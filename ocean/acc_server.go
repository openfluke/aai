package main

import (
	"embed"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/openfluke/tide/report"
)

//go:embed acc.html
var accFS embed.FS

func loadAccHTML() []byte {
	b, err := accFS.ReadFile("acc.html")
	if err != nil {
		return []byte("<h1>acc.html missing</h1>")
	}
	return b
}

type accServer struct {
	addr   string
	title  string
	outDir string
	cache  *cacheIndex
	mu     sync.Mutex
}

func (s *accServer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleAccPage)
	mux.HandleFunc("/acc", s.handleAccPage)
	mux.HandleFunc("/api/acc", s.handleAccJSON)
	mux.HandleFunc("/api/reload", s.handleReload)
	mux.HandleFunc("/compare", s.handleComparePage)
	mux.HandleFunc("/api/compare", s.handleCompareJSON)
	mux.HandleFunc("/api/compare.pdf", s.handleComparePDF)
	mux.HandleFunc("/api/meta", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		farms, cells, root, at := s.cache.Stats()
		_ = json.NewEncoder(w).Encode(map[string]any{
			"title": s.title, "mode": "cache-acc",
			"cache_root": root, "farms": farms, "cells": cells, "loaded_at": at,
			"routes": map[string]string{
				"acc": "/", "acc_api": "/api/acc", "reload": "/api/reload",
				"compare": "/compare", "compare_api": "/api/compare", "compare_pdf": "/api/compare.pdf",
			},
		})
	})
	return mux
}

func (s *accServer) ListenAndServe() error {
	return http.ListenAndServe(s.addr, s.handler())
}

func (s *accServer) handleAccPage(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/acc" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(loadAccHTML())
}

func (s *accServer) handleAccJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(s.cache.Board())
}

func (s *accServer) handleReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "POST or GET", 405)
		return
	}
	s.mu.Lock()
	err := s.cache.reload()
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	farms, cells, root, at := s.cache.Stats()
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ok": true, "farms": farms, "cells": cells, "cache_root": root, "loaded_at": at,
	})
}

func (s *accServer) handleCompareJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(s.cache.Compare(s.title))
}

func (s *accServer) handleComparePDF(w http.ResponseWriter, r *http.Request) {
	rep := s.cache.Compare(s.title)
	pdf, err := report.PDFCompare(rep)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if s.outDir != "" {
		_ = os.MkdirAll(s.outDir, 0o755)
		_ = os.WriteFile(filepath.Join(s.outDir, "ocean-cache-compare.pdf"), pdf, 0o644)
	}
	w.Header().Set("Content-Type", "application/pdf")
	w.Header().Set("Content-Disposition", `attachment; filename="ocean-cache-acc-compare.pdf"`)
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(pdf)
}

func (s *accServer) handleComparePage(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/compare" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write([]byte(compareShimHTML))
}

var compareShimHTML = `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>cache compare</title>
<style>
:root{--bg:#0e1116;--fg:#e8edf5;--muted:#8b95a8;--line:#243044;--accent:#6ec8ff}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--accent)}header{padding:16px 20px;border-bottom:1px solid var(--line)}
main{padding:16px 20px;max-width:1100px}code{background:#1a2230;padding:2px 6px;border-radius:4px}
.note{color:var(--muted);margin-top:8px}
</style></head><body>
<header>
  <strong>Cache compare</strong> · <a href="/">Acc board</a> ·
  <a href="/api/compare">/api/compare JSON</a> · <a href="/api/compare.pdf">PDF</a>
</header>
<main>
  <p>Full overlapping LR×farm charts for this cache are on the <a href="/"><strong>Acc board</strong></a> (pure Acc rank + Acc×Avail + overlapping series).</p>
  <p class="note">This endpoint still builds the classic ocean <code>CompareReport</code> from every <code>results.json</code> under <code>OCEAN_CACHE</code> — open the JSON/PDF for matched recipes, mode×LR heatmaps, and vs-sgd grids.</p>
  <pre id="sum" class="note">loading…</pre>
</main>
<script>
fetch('/api/compare').then(r=>r.json()).then(d=>{
  const el=document.getElementById('sum');
  el.textContent = [
    'machines: '+(d.machines||[]).join(', '),
    'lrs: '+(d.lr_labels||[]).join(', '),
    'summary rows: '+(d.summary||[]).length,
    'scatter: '+(d.scatter||[]).length,
    'mode_series: '+(d.mode_series||[]).length,
    'generated: '+d.generated
  ].join('\n');
}).catch(e=>{document.getElementById('sum').textContent=String(e)});
</script>
</body></html>
`
