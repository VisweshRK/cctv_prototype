package cctv

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

var faceNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 _-]{0,39}$`)
var faceRoot = filepath.Join("ai", "plugins", "face_recognition", "faces")

type facePerson struct {
	Name   string `json:"name"`
	Photos int    `json:"photos"`
}

func validFaceName(name string) bool { return faceNamePattern.MatchString(name) }

func faceGalleryDir() string { return faceRoot }

func faceGallery(rw http.ResponseWriter, r *http.Request) {
	if !localRequest(r) {
		writeJSON(rw, http.StatusForbidden, map[string]string{"error": "face gallery management is available from this computer only"})
		return
	}
	switch r.Method {
	case http.MethodGet:
		listFacePeople(rw)
	case http.MethodPost:
		addFacePhoto(rw, r)
	case http.MethodDelete:
		deleteFacePerson(rw, r)
	default:
		rw.Header().Set("Allow", "GET, POST, DELETE")
		writeJSON(rw, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func listFacePeople(rw http.ResponseWriter) {
	entries, err := os.ReadDir(faceGalleryDir())
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "face gallery unavailable"})
		return
	}
	people := make([]facePerson, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !validFaceName(entry.Name()) {
			continue
		}
		photos, err := os.ReadDir(filepath.Join(faceGalleryDir(), entry.Name()))
		if err != nil {
			continue
		}
		count := 0
		for _, photo := range photos {
			ext := strings.ToLower(filepath.Ext(photo.Name()))
			if !photo.IsDir() && (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
				count++
			}
		}
		people = append(people, facePerson{Name: entry.Name(), Photos: count})
	}
	sort.Slice(people, func(i, j int) bool { return strings.ToLower(people[i].Name) < strings.ToLower(people[j].Name) })
	writeJSON(rw, http.StatusOK, people)
}

func addFacePhoto(rw http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(rw, r.Body, 8<<20)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		writeJSON(rw, http.StatusBadRequest, map[string]string{"error": "upload must be a photo smaller than 8 MB"})
		return
	}
	name := strings.TrimSpace(r.FormValue("name"))
	if !validFaceName(name) {
		writeJSON(rw, http.StatusBadRequest, map[string]string{"error": "name must start with a letter or number and contain only letters, numbers, spaces, _ or - (max 40 characters)"})
		return
	}
	file, _, err := r.FormFile("photo")
	if err != nil {
		writeJSON(rw, http.StatusBadRequest, map[string]string{"error": "photo field is required"})
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, (8<<20)+1))
	if err != nil || len(data) == 0 || len(data) > 8<<20 {
		writeJSON(rw, http.StatusBadRequest, map[string]string{"error": "invalid photo size"})
		return
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || (format != "jpeg" && format != "png") || config.Width < 40 || config.Height < 40 || config.Width > 10000 || config.Height > 10000 {
		writeJSON(rw, http.StatusBadRequest, map[string]string{"error": "photo must be a valid JPEG or PNG at least 40 by 40 pixels"})
		return
	}
	ext := ".jpg"
	if format == "png" {
		ext = ".png"
	}
	if err = os.MkdirAll(filepath.Join(faceGalleryDir(), name), 0700); err != nil {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "could not save gallery photo"})
		return
	}
	var random [12]byte
	if _, err = rand.Read(random[:]); err != nil {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "could not save gallery photo"})
		return
	}
	path := filepath.Join(faceGalleryDir(), name, hex.EncodeToString(random[:])+ext)
	if err = os.WriteFile(path, data, 0600); err != nil {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "could not save gallery photo"})
		return
	}
	writeJSON(rw, http.StatusCreated, facePerson{Name: name, Photos: countFacePhotos(name)})
}

func countFacePhotos(name string) int {
	items, err := os.ReadDir(filepath.Join(faceGalleryDir(), name))
	if err != nil {
		return 0
	}
	count := 0
	for _, item := range items {
		ext := strings.ToLower(filepath.Ext(item.Name()))
		if !item.IsDir() && (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
			count++
		}
	}
	return count
}

func faceGalleryBuildCommand(repoRoot string) (*exec.Cmd, error) {
	root, err := filepath.Abs(repoRoot)
	if err != nil {
		return nil, err
	}
	aiDir := filepath.Join(root, "ai")
	python := filepath.Join(aiDir, ".venv", "Scripts", "python.exe")
	if _, err := os.Stat(python); err != nil {
		python = "python"
	}
	cmd := exec.Command(python, "-m", "plugins.face_recognition.build_gallery")
	cmd.Dir = aiDir
	return cmd, nil
}

func deleteFacePerson(rw http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.URL.Query().Get("name"))
	if !validFaceName(name) {
		writeJSON(rw, http.StatusBadRequest, map[string]string{"error": "invalid person name"})
		return
	}
	path := filepath.Join(faceGalleryDir(), name)
	if info, err := os.Stat(path); err != nil || !info.IsDir() {
		writeJSON(rw, http.StatusNotFound, map[string]string{"error": "person not found"})
		return
	}
	if err := os.RemoveAll(path); err != nil {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "could not remove person"})
		return
	}
	writeJSON(rw, http.StatusOK, map[string]bool{"ok": true})
}

func rebuildFaceGallery(rw http.ResponseWriter, r *http.Request) {
	if !localRequest(r) {
		writeJSON(rw, http.StatusForbidden, map[string]string{"error": "face gallery management is available from this computer only"})
		return
	}
	if r.Method != http.MethodPost {
		rw.Header().Set("Allow", "POST")
		writeJSON(rw, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	root, err := os.Getwd()
	if err != nil {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "could not locate the application directory"})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	cmd, err := faceGalleryBuildCommand(root)
	if err != nil {
		writeJSON(rw, http.StatusInternalServerError, map[string]string{"error": "could not prepare the gallery builder"})
		return
	}
	buildPath, buildDir, buildArgs := cmd.Path, cmd.Dir, cmd.Args[1:]
	cmd = exec.CommandContext(ctx, buildPath, buildArgs...)
	cmd.Dir = buildDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if len(message) > 1200 {
			message = message[len(message)-1200:]
		}
		writeJSON(rw, http.StatusBadRequest, map[string]any{"error": "gallery build failed; check model assets and use at least 3 usable photos per person", "detail": message})
		return
	}
	writeJSON(rw, http.StatusOK, map[string]any{"ok": true, "message": strings.TrimSpace(string(output))})
}

func localRequest(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = strings.Trim(r.RemoteAddr, "[]")
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
