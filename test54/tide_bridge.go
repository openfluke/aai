package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/openfluke/tide/dash"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/pulse"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/quant"
)

type tideBridge struct {
	tr   *pulse.Tracker
	srv  *dash.Server
	cams int
}

func startTideBridge(addr string, jobs []Job, lr float64, kinds []CellKind, cams int) *tideBridge {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return nil
	}
	if cams < 1 {
		cams = 1
	}
	listen := normalizeAddr(addr)
	cells := make([]permute.Cell, 0, len(jobs))
	seen := map[string]bool{}
	for _, j := range jobs {
		c := jobToTideCell(j, cams)
		if seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		cells = append(cells, c)
	}
	tr := pulse.New()
	subtitle := fmt.Sprintf("deep dayroute · depth=%d · %d-cam × mode × dtype × lr", EnvInt("TEST54_DEPTH", 4), cams)
	if len(kinds) == 1 {
		subtitle = fmt.Sprintf("deep dayroute · %s ×%d · %d-cam × mode × dtype × lr", kinds[0], EnvInt("TEST54_DEPTH", 4), cams)
	}
	tideID := EnvOr("TEST54_TIDE_ID", tideIDForCams(cams))
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     listen,
		Epoch:    1,
		Task:     "test54-dayroute",
		Subtitle: subtitle,
		LR:       lr,
		ID:       tideID,
	}
	// Auto-start: test54 does not wait for the dash Start button.
	srv.SignalStart()
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintf(os.Stderr, "tide dash: %v\n", err)
		}
	}()
	fmt.Printf("tide  http://<host>:%s  id=%s  cams=%d  (Lucy + /api/report.pdf)\n", portOf(listen), tideID, cams)
	return &tideBridge{tr: tr, srv: srv, cams: cams}
}

func (t *tideBridge) signalStart() {
	if t != nil && t.srv != nil {
		t.srv.SignalStart()
	}
}

func (t *tideBridge) resetPace() {
	if t != nil && t.srv != nil {
		t.srv.ResetPaceAnchor()
	}
}

func (t *tideBridge) setQueue(done, total int, msg string) {
	if t == nil || t.tr == nil {
		return
	}
	if total < 1 {
		total = 1
	}
	if done < 0 {
		done = 0
	}
	left := total - done
	if left < 0 {
		left = 0
	}
	if msg == "" {
		msg = fmt.Sprintf("%d/%d done · %d left", done, total, left)
	}
	t.tr.SetMeta(done, total, done, total, msg)
}

func (t *tideBridge) beginJob(job Job, phase string, done, total int) {
	if t == nil || t.tr == nil {
		return
	}
	left := total - done
	if left < 0 {
		left = 0
	}
	msg := fmt.Sprintf("%s · %s · %d/%d · %d left", phase, job.ID, done, total, left)
	t.tr.SetMeta(done, total, done, total, msg)
	t.tr.Begin(jobToTideCell(job, t.cams), phase)
}

func (t *tideBridge) pulseRunning(r modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	t.tr.Pulse(lucy.Window{}, tideSnap(r), "train")
}

func (t *tideBridge) finishJob(r modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	status, note := "ok", r.Layer
	if r.Err != "" {
		status, note = "fail", r.Err
	}
	t.tr.Finish(status, note, tideSnap(r))
}

// commitJob records a finished job with real wall-clock start/end (multi-worker safe).
func (t *tideBridge) commitJob(j Job, r modeResult, started, ended time.Time) {
	if t == nil || t.tr == nil {
		return
	}
	status, note := "ok", r.Layer
	if r.Err != "" {
		status, note = "fail", r.Err
	}
	t.tr.Commit(jobToTideCell(j, t.cams), 1, status, note, tideSnap(r), started, ended)
}

func (t *tideBridge) seedCompleted(rows []modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	n := 0
	now := time.Now()
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		// Seeded rows have no wall duration — Commit with equal times so they
		// are ignored by pace (minPaceSec filter).
		t.tr.Commit(resultToTideCell(r, t.cams), 1, "ok", r.Layer, tideSnap(r), now, now)
		n++
	}
	if n > 0 {
		fmt.Printf("tide: seeded %d completed cell(s)\n", n)
	}
}

// tideSnap sends the full Lucy snapshot (AccPerSec / time-to-50 / windows).
// Stripped Acc/Thru-only snaps left LEARN SPEED empty on the Tide board.
func tideSnap(r modeResult) lucy.Snapshot {
	snap := r.Lucy
	if r.Acc > 0 || snap.AvgAccuracy == 0 {
		snap.AvgAccuracy = r.Acc
	}
	if r.Soft > 0 {
		snap.SoftAcc = r.Soft
	}
	if r.Avail > 0 {
		snap.Availability = r.Avail
	}
	if r.Thru > 0 {
		snap.Throughput = r.Thru
	}
	if r.Score > 0 {
		snap.Score = r.Score
	}
	if r.Actions > 0 {
		snap.TotalOutputs = r.Actions
	}
	if r.RAMKiB > 0 {
		snap.WeightBytes = int64(r.RAMKiB * 1024)
	}
	if snap.Duration > 0 {
		snap.AccPerSec = snap.AvgAccuracy / snap.Duration.Seconds()
		mb := float64(snap.WeightBytes) / (1024 * 1024)
		if mb < 1e-9 {
			mb = 1e-9
		}
		snap.MobileAccPerSec = snap.AccPerSec / mb
	}
	snap.Windows = nil
	snap.SoftAccBlocks = nil
	snap.PhaseBlocks = nil
	snap.SwitchBlocks = nil
	return snap
}

func jobToTideCell(j Job, cams int) permute.Cell {
	if cams < 1 {
		cams = 1
	}
	return permute.Cell{
		ID: j.ID, DType: j.DType, Format: quant.FormatNone,
		Mode: welvetModeToTide(j.Mode), Arch: permute.ArchForCams(cams), Cams: cams,
		Backend: core.BackendCPUTiled, UseSIMD: j.Kind == KindDense && j.DType == core.DTypeFloat32,
	}
}

func resultToTideCell(r modeResult, cams int) permute.Cell {
	if cams < 1 {
		cams = 1
	}
	dt := core.ParseDType(r.DType)
	mode, err := parallel.ParseTrainMode(r.Mode)
	if err != nil {
		mode = parallel.ModeNormalBP
	}
	return permute.Cell{
		ID: r.ID, DType: dt, Format: quant.FormatNone,
		Mode: welvetModeToTide(mode), Arch: permute.ArchForCams(cams), Cams: cams,
		Backend: core.BackendCPUTiled,
	}
}

func welvetModeToTide(m parallel.TrainMode) permute.TrainMode {
	switch m {
	case parallel.ModeNormalBP:
		return permute.ModeSGD
	case parallel.ModeStepBP:
		return permute.ModeStepSGD
	case parallel.ModeTween:
		return permute.ModeTween
	case parallel.ModeTweenChain:
		return permute.ModeTweenChain
	case parallel.ModeStepTween:
		return permute.ModeStepTween
	case parallel.ModeStepTweenChain:
		return permute.ModeStepTweenChain
	default:
		return permute.TrainMode(m.String())
	}
}

func normalizeAddr(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return "0.0.0.0:8080"
	}
	if strings.HasPrefix(addr, ":") {
		return "0.0.0.0" + addr
	}
	if !strings.Contains(addr, ":") {
		return "0.0.0.0:" + addr
	}
	return addr
}

func portOf(listen string) string {
	if i := strings.LastIndex(listen, ":"); i >= 0 {
		return listen[i+1:]
	}
	return "8080"
}
