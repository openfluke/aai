package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Store struct {
	mu   sync.Mutex
	Root string
}

type Progress struct {
	Version   int       `json:"version"`
	UpdatedAt time.Time `json:"updated_at"`
	Total     int       `json:"total"`
	NextIndex int       `json:"next_index"`
	DoneIDs   []string  `json:"done_ids"`
	BestID    string    `json:"best_id,omitempty"`
	BestScore float64   `json:"best_score,omitempty"`
	BestAcc   float64   `json:"best_acc,omitempty"`
	Current   string    `json:"current,omitempty"`
}

func NewStore(root string) *Store {
	if root == "" {
		root = "test53_ckpt"
	}
	return &Store{Root: root}
}

func (s *Store) ensure() error { return os.MkdirAll(s.Root, 0o755) }

func (s *Store) progressPath() string { return filepath.Join(s.Root, "progress.json") }
func (s *Store) historyPath() string  { return filepath.Join(s.Root, "history.json") }
func (s *Store) resultsPath() string  { return filepath.Join(s.Root, "results.json") }
func (s *Store) lpdPath() string      { return filepath.Join(s.Root, "lpd.json") }

func (s *Store) Load() (*Progress, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return nil, err
	}
	var p Progress
	if b, err := os.ReadFile(s.progressPath()); err == nil {
		if err := json.Unmarshal(b, &p); err != nil {
			return nil, err
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	if p.Version == 0 {
		p.Version = 1
	}
	return &p, nil
}

func (s *Store) SaveProgress(p *Progress) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return err
	}
	p.UpdatedAt = time.Now().UTC()
	p.Version = 1
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.progressPath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.progressPath())
}

func (s *Store) AppendHistory(h map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return err
	}
	var hist []map[string]any
	if b, err := os.ReadFile(s.historyPath()); err == nil {
		_ = json.Unmarshal(b, &hist)
	}
	hist = append(hist, h)
	b, err := json.MarshalIndent(hist, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.historyPath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.historyPath())
}

func (s *Store) SaveJSON(name string, v any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(s.Root, name)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (s *Store) LoadResults() ([]modeResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var wrap struct {
		Results []modeResult `json:"results"`
	}
	b, err := os.ReadFile(s.resultsPath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if err := json.Unmarshal(b, &wrap); err != nil {
		return nil, err
	}
	return wrap.Results, nil
}

func doneSet(ids []string) map[string]bool {
	m := make(map[string]bool, len(ids))
	for _, id := range ids {
		m[id] = true
	}
	return m
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
