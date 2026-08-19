package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
)

func modeLabel(s string) string {
	return parallel.ShortTrainMode(s)
}

func simdTag(on bool) string {
	if on {
		return "on"
	}
	return "off"
}

func printAccTable(rows []result, nWin int, phase, win time.Duration) {
	pWin := int(phase / win)
	if pWin < 1 {
		pWin = 1
	}
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  ACCURACY OVER TIME (hard Acc % per window)                              ║")
	fmt.Printf("║  [%s CHASE] │ [%s AVOID] │ [%s CHASE]\n", phase.Truncate(time.Millisecond), phase.Truncate(time.Millisecond), phase.Truncate(time.Millisecond))
	fmt.Print("╠════════════════════╦════")
	for i := 0; i < nWin; i++ {
		fmt.Print("╦════")
	}
	fmt.Println("╣")
	fmt.Printf("║ %-18s ║SIMD", "Mode")
	for i := 0; i < nWin; i++ {
		fmt.Printf("║%3d ", i+1)
	}
	fmt.Println("║")
	fmt.Print("╠════════════════════╬════")
	for i := 0; i < nWin; i++ {
		fmt.Print("╬════")
	}
	fmt.Println("╣")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("║ %-18s ║%3s ", clip(modeLabel(r.Mode), 18), simdTag(r.SIMD))
		for i := 0; i < nWin; i++ {
			v := 0.0
			if i < len(r.Acc1s) {
				v = r.Acc1s[i]
			}
			fmt.Printf("║%3.0f%%", v)
		}
		fmt.Println("║")
	}
	fmt.Print("╚════════════════════╩════")
	for i := 0; i < nWin; i++ {
		fmt.Print("╩════")
	}
	fmt.Println("╝")
	fmt.Printf("                         ↑ %s  ↑ %s\n\n", phase, 2*phase)
}

func printAdaptSummary(rows []result, nWin int, phase, win time.Duration) {
	pWin := int(phase / win)
	if pWin < 1 {
		pWin = 1
	}
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  ADAPTATION SUMMARY                                                      ║")
	fmt.Println("╠════════════════════╦════╦══════════╦══════════════╦══════════════╦═══════╣")
	fmt.Println("║ Mode               ║SIMD║ Outputs  ║ 1st flip     ║ 2nd flip     ║ Avg   ║")
	fmt.Println("╠════════════════════╬════╬══════════╬══════════════╬══════════════╬═══════╣")
	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("║ %-18s ║%3s ║ ERR\n", clip(modeLabel(r.Mode), 18), simdTag(r.SIMD))
			continue
		}
		a1 := flipLine(r.Acc1s, pWin-1, pWin, 2*pWin, win)
		a2 := flipLine(r.Acc1s, 2*pWin-1, 2*pWin, nWin, win)
		fmt.Printf("║ %-18s ║%3s ║%9d ║ %-12s ║ %-12s ║%6.1f ║\n",
			clip(modeLabel(r.Mode), 18), simdTag(r.SIMD), r.Outputs, a1, a2, r.Lucy.AvgAccuracy)
	}
	fmt.Println("╚════════════════════╩════╩══════════╩══════════════╩══════════════╩═══════╝")
	fmt.Println()
}

func flipLine(acc []float64, before, from, until int, win time.Duration) string {
	b := 0.0
	if before >= 0 && before < len(acc) {
		b = acc[before]
	}
	d, after := delay50(acc, from, until)
	if d < 0 {
		return fmt.Sprintf("%.0f%%→ —", b)
	}
	return fmt.Sprintf("%.0f%%→%.0f%% %s", b, after, time.Duration(d)*win)
}

func delay50(acc []float64, from, until int) (delay int, val float64) {
	if from < 0 {
		from = 0
	}
	if until > len(acc) {
		until = len(acc)
	}
	for i := from; i < until; i++ {
		if acc[i] >= 50 {
			return i - from, acc[i]
		}
	}
	return -1, 0
}

func printOps(rows []result) {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  OPERATIONAL METRICS  Score = Thru × Avail × Acc / 10,000                ║")
	fmt.Println("╠════════════════════╦════╦════════╦════════╦════════╦════════╦════════════╣")
	fmt.Println("║ Mode               ║SIMD║  Thru  ║ Avail  ║  Acc   ║ Soft   ║ Score      ║")
	fmt.Println("╠════════════════════╬════╬════════╬════════╬════════╬════════╬════════════╣")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		s := r.Lucy
		fmt.Printf("║ %-18s ║%3s ║%7.0f ║%6.1f%% ║%6.1f ║%6.1f ║%11.1f ║\n",
			clip(modeLabel(r.Mode), 18), simdTag(r.SIMD), s.Throughput, s.Availability, s.AvgAccuracy, s.SoftAcc, s.Score)
	}
	fmt.Println("╚════════════════════╩════╩════════╩════════╩════════╩════════╩════════════╝")
	fmt.Println()
}

func printLPD(rows []result) {
	var pts []lucy.Sample
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		pts = append(pts, lucy.Sample{
			ID:     r.Mode + "/" + simdTag(r.SIMD),
			Mode:   r.Mode,
			DType:  "float32",
			Arch:   "dense×6",
			Score:  r.Lucy.Score,
			Soft:   r.Lucy.SoftAcc,
			Acc:    r.Lucy.AvgAccuracy,
			Thru:   r.Lucy.Throughput,
			Avail:  r.Lucy.Availability,
			RAMKiB: r.RAMKiB,
		})
	}
	l := lucy.BuildLPD(pts)
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  LUCY DENSITY (LPD)                                                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println(l.Formula)
	if l.AccChamp.ID != "" {
		fmt.Printf("Acc champ     %s   Acc %.1f  Thru %.0f  Avail %.1f%%  %.1f KiB\n",
			modeLabel(l.AccChamp.ID), l.AccChamp.Acc, l.AccChamp.Thru, l.AccChamp.Avail, l.AccChamp.RAMKiB)
	}
	if l.Champ.ID != "" {
		fmt.Printf("Score champ   %s   Score %.1f  Acc %.1f\n", modeLabel(l.Champ.ID), l.Champ.Score, l.Champ.Acc)
	}
	if l.LiveChamp.ID != "" {
		fmt.Printf("Live-fit (Q)  %s\n", modeLabel(l.LiveChamp.ID))
	}
	if l.GoldStd.ID != "" {
		fmt.Printf("Gold-std      %s  mode %s  Acc %.1f  LPD %.2f\n", modeLabel(l.GoldStd.ID), modeLabel(l.GoldStd.Mode), l.GoldStd.Acc, l.GoldStd.LPD)
	}
	fmt.Println()
	fmt.Println("╠ Band  Mode/SIMD              Acc%  Thru%  Avail%   Q%   LPD    Acc   Thru  KiB")
	show := append([]lucy.LPDRow{}, l.Top...)
	if len(show) > 16 {
		show = show[:16]
	}
	for _, row := range show {
		fmt.Printf("║ %-5s %-22s %5.0f %6.0f %7.0f %5.0f %6.2f %6.1f %6.0f %5.1f\n",
			row.Band, clip(modeLabel(row.ID), 22), row.RelAcc*100, row.RelThru*100, row.RelAvail*100,
			row.Q*100, row.LPD, row.Acc, row.Thru, row.RAMKiB)
	}
	fmt.Println()
}

func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "~"
}
