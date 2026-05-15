#!/usr/bin/env bash
# Benchmark all 5 v7 variants. Outputs CSV to stdout.
set -u

BINARIES=(v7_base v7_forced v7_parity v7_level4 v7_ns4)
SAMPLES="${SAMPLES:-5}"
TIMEOUT="${TIMEOUT:-30}"
AB_LIMIT="${AB_LIMIT:-1000000}"

# (n, tuple) test cases
TESTS=("30 0" "30 1" "33 0" "33 1" "34 0")

echo "binary,n,tuple,sample,real_secs,cd_total,cd_passed,solved"

for tc in "${TESTS[@]}"; do
    read -r n tuple <<<"$tc"
    for bin in "${BINARIES[@]}"; do
        for s in $(seq 1 "$SAMPLES"); do
            rm -f "BS_$((n+1))_${n}_V7Parallel_tuple${tuple}_"*.txt 2>/dev/null
            out=$(/usr/bin/time -p "./target/release/$bin" "$n" --tuple "$tuple" \
                  --timeout "$TIMEOUT" --ab-limit "$AB_LIMIT" 2>&1)
            real=$(echo "$out" | grep '^real ' | awk '{print $2}')
            # Solution path uses "CD pairs tried"; timeout path uses "CD pairs checked" + "...searched (passed spectral)".
            if echo "$out" | grep -q 'Solution at tuple'; then
                cd_total=$(echo "$out" | grep -oE 'CD pairs tried: [0-9.e+]+' | awk '{print $NF}')
                cd_passed="$cd_total"
                solved=1
            else
                cd_total=$(echo "$out" | grep -oE 'CD pairs checked: [0-9.e+]+' | awk '{print $NF}')
                cd_passed=$(echo "$out" | grep -oE 'CD pairs searched \(passed spectral\): [0-9.e+]+' | awk '{print $NF}')
                solved=0
            fi
            echo "$bin,$n,$tuple,$s,$real,${cd_total:-NA},${cd_passed:-NA},$solved"
        done
    done
done
rm -f BS_*_tuple*.txt 2>/dev/null
