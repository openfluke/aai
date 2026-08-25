package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/openfluke/tide/dash"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/pulse"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/quant"
)

type tideBridge struct {
	tr  *pulse.Tracker
	srv *dash.Server
}

func startTideBridge(addr string, jobs []Job, lr float64) *tideBridge {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return nil
	}
	listen := normalizeAddr(addr)
	cells := make([]permute.Cell, 0, len(jobs))
	seen := map[string]bool{}
	for _, j := range jobs {
		c := jobToTideCell(j)
		if seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		cells = append(cells, c)
	}
	tr := pulse.New()
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     listen,
		Epoch:    1,
		Task:     "test53-dayroute",
		Subtitle: "dayroute · 5-day XY life schedule × layer × mode × dtype",
		LR:       lr,
		ID:       "test53",
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintf(os.Stderr, "tide dash: %v\n", err)
		}
	}()
	fmt.Printf("tide  http://<host>:%s  (Lucy + /api/report.pdf)\n", portOf(listen))
	return &tideBridge{tr: tr, srv: srv}
}

func (t *tideBridge) signalStart() {
	if t != nil && t.srv != nil {
		t.srv.SignalStart()
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
	t.tr.Begin(jobToTideCell(job), phase)
}

func (t *tideBridge) pulseRunning(r modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	t.tr.Pulse(lucy.Window{}, lucy.Snapshot{
		AvgAccuracy: r.Acc, SoftAcc: r.Soft, Availability: r.Avail,
		Throughput: r.Thru, Score: r.Score, TotalOutputs: r.Actions,
		WeightBytes: int64(r.RAMKiB * 1024),
	}, "train")
}

func (t *tideBridge) finishJob(r modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	status, note := "ok", r.Layer
	if r.Err != "" {
		status, note = "fail", r.Err
	}
	t.tr.Finish(status, note, lucy.Snapshot{
		AvgAccuracy: r.Acc, SoftAcc: r.Soft, Availability: r.Avail,
		Throughput: r.Thru, Score: r.Score, TotalOutputs: r.Actions,
		WeightBytes: int64(r.RAMKiB * 1024),
	})
}

func (t *tideBridge) seedCompleted(rows []modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	n := 0
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		t.tr.Begin(resultToTideCell(r), "train")
		t.tr.Finish("ok", r.Layer, lucy.Snapshot{
			AvgAccuracy: r.Acc, SoftAcc: r.Soft, Availability: r.Avail,
			Throughput: r.Thru, Score: r.Score, TotalOutputs: r.Actions,
			WeightBytes: int64(r.RAMKiB * 1024),
		})
		n++
	}
	if n > 0 {
		fmt.Printf("tide: seeded %d completed cell(s)\n", n)
	}
}

func jobToTideCell(j Job) permute.Cell {
	return permute.Cell{
		ID: j.ID, DType: j.DType, Format: quant.FormatNone,
		Mode: welvetModeToTide(j.Mode), Arch: permute.ArchForCams(1), Cams: 1,
		Backend: core.BackendCPUTiled, UseSIMD: j.Kind == KindDense && j.DType == core.DTypeFloat32,
	}
}

func resultToTideCell(r modeResult) permute.Cell {
	dt := core.ParseDType(r.DType)
	mode, err := parallel.ParseTrainMode(r.Mode)
	if err != nil {
		mode = parallel.ModeNormalBP
	}
	return permute.Cell{
		ID: r.ID, DType: dt, Format: quant.FormatNone,
		Mode: welvetModeToTide(mode), Arch: permute.ArchForCams(1), Cams: 1,
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
