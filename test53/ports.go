package main

import (
	"fmt"
	"strconv"
	"strings"
)

// tidePortForCam maps cam count × LR band to a host Tide port.
// cam1 lo=8080 neg=8081 hi=8082; cam3 lo=8100 neg=8101 hi=8102; etc.
func tidePortForCam(cam int, band string) int {
	if cam < 1 {
		cam = 1
	}
	off := 0
	switch strings.ToLower(strings.TrimSpace(band)) {
	case "neg", "negative":
		off = 1
	case "hi", "high", "extreme":
		off = 2
	}
	return 8080 + (cam-1)*10 + off
}

func parseCams(spec string) (int, error) {
	spec = strings.TrimSpace(strings.ToLower(spec))
	if spec == "" || spec == "1" || spec == "single" {
		return 1, nil
	}
	if strings.HasPrefix(spec, "cam") {
		spec = strings.TrimPrefix(spec, "cam")
	}
	if strings.HasSuffix(spec, "cam") {
		spec = strings.TrimSuffix(spec, "cam")
	}
	n, err := strconv.Atoi(spec)
	if err != nil || n < 1 || n > 256 {
		return 0, fmt.Errorf("cams must be 1–256, got %q", spec)
	}
	return n, nil
}

func tideIDForCams(cams int) string {
	if cams <= 1 {
		return "test53"
	}
	return fmt.Sprintf("test53-cam%d", cams)
}
