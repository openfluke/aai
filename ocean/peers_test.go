package main

import "testing"

func TestExpandCamPeers(t *testing.T) {
	got := expandCamPeers("192.168.0.22", "1,3", "lo,hi")
	if len(got) != 4 {
		t.Fatalf("want 4 peers got %d", len(got))
	}
	want := map[string]string{
		"cam1-lo": "http://192.168.0.22:8080",
		"cam3-lo": "http://192.168.0.22:8100",
		"cam1-hi": "http://192.168.0.22:8082",
		"cam3-hi": "http://192.168.0.22:8102",
	}
	for _, p := range got {
		if want[p.Name] != p.URL {
			t.Errorf("%s: got %s want %s", p.Name, p.URL, want[p.Name])
		}
	}
}

func TestMergePeersExplicitWins(t *testing.T) {
	explicit := parsePeers("cam1-lo=http://10.0.0.1:8080")
	generated := expandCamPeers("192.168.0.22", "1,3", "lo")
	out := mergePeers(explicit, generated)
	byName := map[string]string{}
	for _, p := range out {
		byName[p.Name] = p.URL
	}
	if byName["cam1-lo"] != "http://10.0.0.1:8080" {
		t.Fatalf("explicit cam1-lo should win: %v", byName["cam1-lo"])
	}
	if byName["cam3-lo"] != "http://192.168.0.22:8100" {
		t.Fatalf("generated cam3-lo missing: %v", byName)
	}
}
