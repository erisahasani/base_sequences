#!/usr/bin/env bash
# Roll the per-node perftest logs into ONE shareable summary file.
#   Usage: ./analyze_perftest.sh <LOG_DIR>
# Writes <LOG_DIR>/perftest_summary.txt (and prints it). Share that one file.
set -euo pipefail
DIR=${1:?usage: analyze_perftest.sh <LOG_DIR>}
shopt -s nullglob
outs=("$DIR"/node*.out)
[ ${#outs[@]} -gt 0 ] || { echo "no node*.out files in $DIR"; exit 1; }

SUMMARY="$DIR/perftest_summary.txt"
# PERF value. `|| true`: a node with no PERF block (crash/kill) must not abort the
# whole summary under `set -e`/pipefail — it just yields empty and prints as "?".
pv() { { grep -h "^PERF $1:" "$2" || true; } | tail -1 | sed "s/^PERF $1: *//"; }
# Actual rayon worker-thread count from a node log. This is the AUTHORITATIVE core
# check: the `machine_node*.txt` `nproc` can read 1 under a restricted probe cpuset
# even when the workers ran fully parallel (a known false alarm) — this line is what
# the binary actually spun up, so trust it over nproc.
threads_of() { { grep -hoE 'rayon pool = [0-9]+' "$1" || true; } | grep -oE '[0-9]+' | tail -1; }

{
  echo "================ PERFTEST SUMMARY ================"
  echo "generated: $(date -Is)"
  echo "log_dir:   $DIR"
  echo

  echo "---- run config ----"
  cat "$DIR/run_config.txt" 2>/dev/null || echo "(run_config.txt missing)"
  echo
  echo "---- build ----"
  cat "$DIR/build_info.txt" 2>/dev/null || echo "(build_info.txt missing)"
  echo

  echo "---- machine (per-node; flags heterogeneity) ----"
  cat "$DIR"/machine_node*.txt 2>/dev/null | sort -V || echo "(no machine info)"
  echo "distinct CPU models: $(cat "$DIR"/machine_node*.txt 2>/dev/null | sed -n 's/.*cpu="\([^"]*\)".*/\1/p' | sort -u | wc -l)"
  echo "nodes reporting avx512=yes: $(grep -l 'avx512=yes' "$DIR"/machine_node*.txt 2>/dev/null | wc -l)/${#outs[@]}"
  echo "NOTE: machine 'nproc' above is a probe artifact; trust 'threads' below (rayon pool)."
  echo

  echo "---- per-node throughput ----"
  printf "%-5s | %-8s | %-12s | %-13s | %-12s | %-10s | %s\n" node threads cd_checked cd_rate/s cd_searched ab_to elapsed_s
  : > /tmp/_pt_checked
  : > /tmp/_pt_searched
  : > /tmp/_pt_threads
  for f in $(printf '%s\n' "${outs[@]}" | sort -V); do
    node=$(basename "$f" .out | sed 's/[^0-9]//g')
    thr=$(threads_of "$f")
    chk=$(pv cd_checked "$f"); rate=$(pv cd_rate_per_s "$f")
    srch=$(pv cd_searched "$f"); el=$(pv elapsed_s "$f")
    abto=$(pv ab_timeouts "$f")
    printf "%-5s | %-8s | %-12s | %-13s | %-12s | %-10s | %s\n" \
      "$node" "${thr:-?}" "${chk:-?}" "${rate:-?}" "${srch:-?}" "${abto:-?}" "${el:-?}"
    [ -n "$chk" ] && echo "$chk" >> /tmp/_pt_checked
    [ -n "$srch" ] && echo "$srch" >> /tmp/_pt_searched
    [ -n "$thr" ] && echo "$thr" >> /tmp/_pt_threads
  done
  echo

  echo "---- threads (rayon workers per node) ----"
  # The single most-misread number in past runs. Spell it out unambiguously.
  if [ -s /tmp/_pt_threads ]; then
    echo "per-node worker threads: $(sort -n /tmp/_pt_threads | uniq -c | awk '{printf "%dx%s ", $1, $2}')"
    min_thr=$(sort -n /tmp/_pt_threads | head -1)
    if [ "${min_thr:-0}" -le 1 ]; then
      echo "!! WARNING: a node ran with <=1 worker thread -- single-threaded. Check -c on srun."
    else
      echo "OK: all reporting nodes ran multi-threaded (>=2)."
    fi
  else
    echo "(no 'rayon pool = N' lines found in node*.out -- cannot confirm threading)"
  fi
  echo

  echo "---- aggregate + balance ----"
  awk '{v[NR]=$1; s+=$1; if(NR==1||$1>mx)mx=$1; if(NR==1||$1<mn)mn=$1}
    END{
      if(NR==0){print "no PERF cd_checked values (did nodes reach --timeout?)"; exit}
      mean=s/NR; for(i=1;i<=NR;i++){d=v[i]-mean; ss+=d*d}
      printf "nodes reporting   : %d\n", NR
      printf "aggregate checked : %.4e CD pairs (sum over nodes)\n", s
      printf "per-node mean     : %.4e\n", mean
      printf "balance min/max   : %.4e / %.4e\n", mn, mx
      printf "balance max/mean  : %.3f   spread cv : %.1f%%   (H1 target: well under 10%%)\n", mx/mean, 100*sqrt(ss/NR)/mean
    }' /tmp/_pt_checked
  # cd_searched aggregate + overall spectral pass rate (searched/checked).
  awk 'NR==FNR{c+=$1; next}{t+=$1} END{
      if(c>0) printf "aggregate searched: %.4e (passed spectral) -- pass rate %.3f%%\n", t, 100*t/c;
    }' /tmp/_pt_checked /tmp/_pt_searched
  echo "m3xm6 pairs (PERF mod6, per node): $(pv mod6 "${outs[0]}")"
  echo

  echo "---- stability ----"
  echo "nodes that hit TIMEOUT cleanly: $(grep -l 'TIMEOUT - no solution' "${outs[@]}" 2>/dev/null | wc -l)/${#outs[@]}"
  echo "coverage markers (expect 0): $(ls "$DIR"/cov 2>/dev/null | wc -l)"
  echo "claim/partition I/O error lines: $(grep -h 'non-EEXIST claim' "${outs[@]}" 2>/dev/null | wc -l)"
  echo "panics/errors found:"
  grep -hiE 'panic|thread .* panicked|illegal instruction|SIGILL|error:' "${outs[@]}" | sort -u | head || echo "  none"
  echo "=================================================="
} | tee "$SUMMARY"

echo
echo ">>> SHARE: $SUMMARY"
