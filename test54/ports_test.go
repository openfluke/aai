package main

import "testing"

func TestTidePortForCam(t *testing.T) {
	cases := []struct {
		cam  int
		band string
		want int
	}{
		{1, "lo", 8080},
		{1, "neg", 8081},
		{1, "hi", 8082},
		{3, "lo", 8100},
		{3, "neg", 8101},
		{3, "hi", 8102},
	}
	for _, c := range cases {
		got := tidePortForCam(c.cam, c.band)
		if got != c.want {
			t.Errorf("cam=%d band=%s: got %d want %d", c.cam, c.band, got, c.want)
		}
	}
}

func TestParseCams(t *testing.T) {
	for spec, want := range map[string]int{
		"1": 1, "cam3": 3, "3": 3, "": 1,
	} {
		got, err := parseCams(spec)
		if err != nil || got != want {
			t.Errorf("parseCams(%q) = %d, %v; want %d", spec, got, err, want)
		}
	}
}
