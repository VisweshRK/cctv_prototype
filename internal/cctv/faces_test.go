package cctv

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFaceGalleryBuildCommandUsesAbsolutePaths(t *testing.T) {
	root := t.TempDir()
	python := filepath.Join(root, "ai", ".venv", "Scripts", "python.exe")
	require.NoError(t, os.MkdirAll(filepath.Dir(python), 0o755))
	require.NoError(t, os.WriteFile(python, []byte("python"), 0o600))

	cmd, err := faceGalleryBuildCommand(root)
	require.NoError(t, err)
	require.Equal(t, python, cmd.Path)
	require.Equal(t, filepath.Join(root, "ai"), cmd.Dir)
	require.True(t, filepath.IsAbs(cmd.Path))
	require.Equal(t, []string{"-m", "plugins.face_recognition.build_gallery"}, cmd.Args[1:])
}

func TestFaceGalleryNameValidation(t *testing.T) {
	for _, name := range []string{"Alice", "John Doe", "staff_1", "gate-guard"} {
		require.True(t, validFaceName(name))
	}
	for _, name := range []string{"", "../other", ".", "a/b", "name.", " leading", strings.Repeat("a", 41)} {
		require.False(t, validFaceName(name))
	}
}

func TestFaceGalleryUploadListAndDelete(t *testing.T) {
	oldRoot := faceRoot
	faceRoot = t.TempDir()
	t.Cleanup(func() { faceRoot = oldRoot })

	var pngBytes bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 80, 80))
	for y := 0; y < 80; y++ {
		for x := 0; x < 80; x++ {
			img.Set(x, y, color.RGBA{R: 220, A: 255})
		}
	}
	require.NoError(t, png.Encode(&pngBytes, img))

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	require.NoError(t, form.WriteField("name", "Alice Smith"))
	photo, err := form.CreateFormFile("photo", "portrait.png")
	require.NoError(t, err)
	_, err = photo.Write(pngBytes.Bytes())
	require.NoError(t, err)
	require.NoError(t, form.Close())

	req := httptest.NewRequest(http.MethodPost, "/api/cctv/faces", &body)
	req.RemoteAddr = "127.0.0.1:12345"
	req.Header.Set("Content-Type", form.FormDataContentType())
	w := httptest.NewRecorder()
	faceGallery(w, req)
	require.Equal(t, http.StatusCreated, w.Code, w.Body.String())
	require.Contains(t, w.Body.String(), `"photos":1`)

	w = httptest.NewRecorder()
	getReq := httptest.NewRequest(http.MethodGet, "/api/cctv/faces", nil)
	getReq.RemoteAddr = "127.0.0.1:12345"
	faceGallery(w, getReq)
	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), `"name":"Alice Smith"`)

	req = httptest.NewRequest(http.MethodDelete, "/api/cctv/faces?name=Alice%20Smith", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	w = httptest.NewRecorder()
	faceGallery(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	require.NoDirExists(t, faceRoot+"/Alice Smith")
}

func TestFaceGalleryRejectsInvalidImageAndUnsafeName(t *testing.T) {
	oldRoot := faceRoot
	faceRoot = t.TempDir()
	t.Cleanup(func() { faceRoot = oldRoot })

	for _, name := range []string{"../outside", ""} {
		var body bytes.Buffer
		form := multipart.NewWriter(&body)
		_ = form.WriteField("name", name)
		file, _ := form.CreateFormFile("photo", "x.png")
		_, _ = file.Write([]byte("not an image"))
		_ = form.Close()
		req := httptest.NewRequest(http.MethodPost, "/api/cctv/faces", &body)
		req.RemoteAddr = "127.0.0.1:12345"
		req.Header.Set("Content-Type", form.FormDataContentType())
		w := httptest.NewRecorder()
		faceGallery(w, req)
		require.Equal(t, http.StatusBadRequest, w.Code)
	}
}

func TestFaceGalleryRejectsNonLocalRequests(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/api/cctv/faces", nil)
	r.RemoteAddr = "192.168.1.50:12345"
	faceGallery(w, r)
	require.Equal(t, http.StatusForbidden, w.Code)
}
