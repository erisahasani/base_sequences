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
  echo

  echo "---- per-node throughput ----"
  printf "%-5s | %-12s | %-13s | %-12s | %-10s | %s\n" node cd_checked cd_rate/s cd_searched ab_to elapsed_s
  : > /tmp/_pt_checked
  for f in $(printf '%s\n' "${outs[@]}" | sort -V); do
    node=$(basename "$f" .out | sed 's/[^0-9]//g')
    chk=$(pv cd_checked "$f"); rate=$(pv cd_rate_per_s "$f")
    srch=$(pv cd_searched "$f"); el=$(pv elapsed_s "$f")
    abto=$(pv ab_timeouts "$f")
    printf "%-5s | %-12s | %-13s | %-12s | %-10s | %s\n" \
      "$node" "${chk:-?}" "${rate:-?}" "${srch:-?}" "${abto:-?}" "${el:-?}"
    [ -n "$chk" ] && echo "$chk" >> /tmp/_pt_checked
  done
  echo

  echo "---- aggregate + balance ----"
  awk '{v[NR]=$1; s+=$1; if(NR==1||$1>mx)mx=$1; if(NR==1||$1<mn)mn=$1}
    END{
      if(NR==0){print "no PERF cd_checked values (did nodes reach --timeout?)"; exit}
      mean=s/NR; for(i=1;i<=NR;i++){d=v[i]-mean; ss+=d*d}
      printf "nodes reporting   : %d\n", NR
      printf "aggregate checked : %.4e CD pairs (sum over nodes, ~2h)\n", s
      printf "per-node mean     : %.4e\n", mean
      printf "balance min/max   : %.4e / %.4e\n", mn, mx
      printf "balance max/mean  : %.3f   spread cv : %.1f%%   (want < ~10%%)\n", mx/mean, 100*sqrt(ss/NR)/mean
    }' /tmp/_pt_checked
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
