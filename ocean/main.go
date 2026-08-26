package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/openfluke/tide/ocean"
)

func main() {
	LoadDotEnv(".env")
	// .env peers win over a leftover TIDE_PEERS in the shell (old quick_sprint farms).
	if v := dotenvKey(".env", "TIDE_PEERS"); v != "" {
		_ = os.Setenv("TIDE_PEERS", v)
	}

	addr := flag.String("addr", EnvOr("OCEAN_ADDR", "0.0.0.0:8090"), "ocean listen addr")
	title := flag.String("title", EnvOr("OCEAN_TITLE", "aai ocean"), "dashboard title")
	peersFlag := flag.String("peers", EnvOr("TIDE_PEERS", ""),
		"comma tides: url or name=url (e.g. m4=http://192.168.0.22:8080,m5=http://192.168.0.244:8082)")
	outDir := flag.String("out", EnvOr("OCEAN_OUT", "results"), "optional PDF/chart write dir")
	allowReg := flag.Bool("allow-register", EnvBool("OCEAN_ALLOW_REGISTER", false),
		"allow POST /api/register to add peers (default off — .env list only)")
	flag.Parse()

	peers := parsePeers(*peersFlag)
	if len(peers) == 0 {
		fmt.Fprintln(os.Stderr, "error: set TIDE_PEERS in .env (or -peers)")
		fmt.Fprintln(os.Stderr, "  example: TIDE_PEERS=m4=http://192.168.0.22:8080,m5=http://192.168.0.244:8082")
		os.Exit(2)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}

	fmt.Println("════════════════════════════════════════════════════════")
	fmt.Println(" aai ocean — watch remote tides (no training here)")
	fmt.Printf(" listen: %s  →  http://localhost%s\n", *addr, portOf(*addr))
	fmt.Printf(" title:  %s\n", *title)
	fmt.Printf(" register: %v (static-only=%v)\n", *allowReg, !*allowReg)
	fmt.Println(" peers:")
	for _, p := range peers {
		fmt.Printf("   %-16s %s\n", p.Name, p.URL)
	}
	fmt.Println("════════════════════════════════════════════════════════")

	srv := &ocean.Server{
		Addr:       *addr,
		Title:      *title,
		Peers:      peers,
		OutDir:     *outDir,
		StaticOnly: !*allowReg,
	}
	log.Fatal(srv.ListenAndServe())
}

func portOf(addr string) string {
	if i := strings.LastIndex(addr, ":"); i >= 0 {
		return addr[i:]
	}
	return ":8090"
}

// parsePeers accepts:
//
//	http://host:8080,http://host:8082
//	m4=http://host:8080,m5=http://host:8082
func parsePeers(spec string) []ocean.Peer {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil
	}
	var out []ocean.Peer
	seen := map[string]bool{}
	for i, tok := range strings.FieldsFunc(spec, func(r rune) bool {
		return r == ',' || r == ';' || r == '\n'
	}) {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		name, url := "", tok
		if k, v, ok := strings.Cut(tok, "="); ok {
			name = strings.TrimSpace(k)
			url = strings.TrimSpace(v)
		}
		url = strings.TrimRight(url, "/")
		if url == "" {
			continue
		}
		if name == "" {
			name = fmt.Sprintf("tide-%d", i+1)
		}
		key := strings.ToLower(name)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, ocean.Peer{Name: name, URL: url})
	}
	return out
}
