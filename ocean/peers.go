package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/openfluke/tide/ocean"
	"github.com/openfluke/tide/report"
)

// resolvePeers merges explicit TIDE_PEERS with cam×band peers from TIDE_PEER_HOST + TIDE_CAMS + TIDE_BANDS.
func resolvePeers(explicit string) []ocean.Peer {
	out := parsePeers(explicit)
	host := strings.TrimSpace(EnvOr("TIDE_PEER_HOST", ""))
	camsSpec := strings.TrimSpace(EnvOr("TIDE_CAMS", ""))
	if host == "" || camsSpec == "" {
		return out
	}
	bandsSpec := EnvOr("TIDE_BANDS", "lo,hi")
	generated := expandCamPeers(host, camsSpec, bandsSpec)
	return mergePeers(out, generated)
}

func expandCamPeers(host, camsSpec, bandsSpec string) []ocean.Peer {
	var cams []int
	for _, tok := range strings.FieldsFunc(camsSpec, func(r rune) bool {
		return r == ',' || r == ' ' || r == ';'
	}) {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		if strings.HasPrefix(strings.ToLower(tok), "cam") {
			tok = tok[3:]
		}
		n, err := strconv.Atoi(tok)
		if err != nil || n < 1 {
			continue
		}
		cams = append(cams, n)
	}
	var bands []string
	for _, tok := range strings.FieldsFunc(bandsSpec, func(r rune) bool {
		return r == ',' || r == ' ' || r == ';'
	}) {
		tok = strings.ToLower(strings.TrimSpace(tok))
		if tok != "" {
			bands = append(bands, tok)
		}
	}
	if len(bands) == 0 {
		bands = []string{"lo"}
	}
	host = strings.TrimRight(host, "/")
	var out []ocean.Peer
	for _, cam := range cams {
		for _, band := range bands {
			name := report.CamPeerName(cam, band)
			port := report.TidePortForCam(cam, band)
			out = append(out, ocean.Peer{
				Name: name,
				URL:  fmt.Sprintf("http://%s:%d", host, port),
			})
		}
	}
	return out
}

func mergePeers(primary, extra []ocean.Peer) []ocean.Peer {
	seen := map[string]bool{}
	var out []ocean.Peer
	add := func(p ocean.Peer) {
		key := strings.ToLower(strings.TrimSpace(p.Name))
		if key == "" {
			key = strings.ToLower(p.URL)
		}
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, p)
	}
	for _, p := range primary {
		add(p)
	}
	for _, p := range extra {
		add(p)
	}
	return out
}
