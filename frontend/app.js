(function () {
    "use strict";

    // ---------- exact mirror of vae/trace_encoder.py ----------
    var SEQ_LEN = 300;   // trace_encoder.SEQ_LEN — every trace is padded/truncated to this

    var EVENT_TYPE_VOCAB = {
        "<PAD>": 0, "DEPOSIT_START": 1, "DEPOSIT_END": 2, "WITHDRAW_START": 3,
        "WITHDRAW_END": 4, "LOCK_ACQUIRE_ATTEMPT": 5, "LOCK_ACQUIRED": 6,
        "LOCK_RELEASED": 7, "READ_BALANCE": 8, "WRITE_BALANCE": 9,
        "DEADLOCK_DETECTED": 10, "TRANSFER_COMPLETE": 11
    };

    function buildThreadIndex(events) {
        var mapping = {}, next = 1;
        for (var i = 0; i < events.length; i++) {
            var name = events[i].thread_name;
            if (!(name in mapping)) {
                if (name.toLowerCase().indexOf("main") !== -1) mapping[name] = 0;
                else { mapping[name] = next; next++; }
            }
        }
        return mapping;
    }

    function safeLogValue(v) {
        if (v === null || v === undefined || v === 0) return 0.0;
        var sign = v > 0 ? 1 : -1;
        return sign * Math.log1p(Math.abs(v));
    }

    function encodeRealEvents(allEvents) {
        // Python truncates to SEQ_LEN *before* building the thread map and the
        // delta normaliser, so both must be computed on the same prefix or the
        // rows shown here would not be the rows the encoder actually produces.
        var events = allEvents.slice(0, SEQ_LEN);
        var n = events.length;
        var threadMap = buildThreadIndex(events);
        var ts = events.map(function (e) { return e.timestamp_ns; });
        var deltaNs = [0];
        for (var i = 1; i < n; i++) deltaNs.push(ts[i] - ts[i - 1]);
        var deltaLog = deltaNs.map(function (d) { return Math.log1p(Math.max(d, 0) / 1e6); });
        var maxDelta = deltaLog.length ? Math.max.apply(null, deltaLog) : 1;
        if (maxDelta === 0) maxDelta = 1;
        var rows = [];
        for (var j = 0; j < n; j++) {
            var e = events[j];
            var rawVal = (e.value === undefined) ? null : e.value;
            rows.push({
                seq: e.seq, thread_name: e.thread_name, event_type: e.event_type,
                event_type_id: EVENT_TYPE_VOCAB.hasOwnProperty(e.event_type) ? EVENT_TYPE_VOCAB[e.event_type] : 0,
                thread_index: threadMap.hasOwnProperty(e.thread_name) ? threadMap[e.thread_name] : 0,
                value_normalized: safeLogValue(rawVal),
                has_value: rawVal === null ? 0 : 1,
                delta_time_normalized: deltaLog[j] / maxDelta,
                timestamp_ns: e.timestamp_ns
            });
        }
        return {
            rows: rows,
            threadMap: threadMap,
            encodedEvents: events,
            droppedByTruncation: allEvents.length - n
        };
    }

    function lockHealth(events) {
        var byThread = {};
        events.forEach(function (e) {
            var t = e.thread_name;
            if (!byThread[t]) byThread[t] = { attempts: 0, acquired: 0, released: 0 };
            var d = byThread[t];
            if (e.event_type === "LOCK_ACQUIRE_ATTEMPT") d.attempts++;
            else if (e.event_type === "LOCK_ACQUIRED") d.acquired++;
            else if (e.event_type === "LOCK_RELEASED") d.released++;
        });
        Object.keys(byThread).forEach(function (t) { byThread[t].unmatched = byThread[t].attempts - byThread[t].acquired; });
        return byThread;
    }

    // ---------- embedded compact samples ----------
    var SAMPLES = {
        deadlock: {
            run_id: 1, variant: "deadlock", initial_balance: 1000.0, expected_balance: null,
            actual_balance: null, correct: null, deadlock_detected: true,
            events: [
                { seq: 1, timestamp_ns: 2269914288600, thread_name: "Deadlock-Run01-AtoB", event_type: "DEPOSIT_START", value: 100.0 },
                { seq: 2, timestamp_ns: 2269914306061, thread_name: "Deadlock-Run01-AtoB", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 3, timestamp_ns: 2269914313463, thread_name: "Deadlock-Run01-AtoB", event_type: "LOCK_ACQUIRED", value: null },
                { seq: 4, timestamp_ns: 2269914888980, thread_name: "Deadlock-Run01-BtoA", event_type: "DEPOSIT_START", value: 200.0 },
                { seq: 5, timestamp_ns: 2269914905233, thread_name: "Deadlock-Run01-BtoA", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 6, timestamp_ns: 2269914912711, thread_name: "Deadlock-Run01-BtoA", event_type: "LOCK_ACQUIRED", value: null },
                { seq: 7, timestamp_ns: 2270014709551, thread_name: "Deadlock-Run01-AtoB", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 8, timestamp_ns: 2270023748576, thread_name: "Deadlock-Run01-BtoA", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 9, timestamp_ns: 2273917538218, thread_name: "MainThread", event_type: "DEADLOCK_DETECTED", value: null }
            ]
        },
        race: {
            run_id: 1, variant: "race_condition", initial_balance: 1000.0, expected_balance: 776.52,
            actual_balance: 948.57, correct: false, deadlock_detected: false, truncated: true,
            events: [
                { seq: 1, timestamp_ns: 2262954371888, thread_name: "Race-Run01-T1", event_type: "DEPOSIT_START", value: 91.72 },
                { seq: 2, timestamp_ns: 2262954379736, thread_name: "Race-Run01-T1", event_type: "READ_BALANCE", value: 1000.0 },
                { seq: 3, timestamp_ns: 2262954778167, thread_name: "Race-Run01-T2", event_type: "WITHDRAW_START", value: 91.5 },
                { seq: 4, timestamp_ns: 2262954788073, thread_name: "Race-Run01-T2", event_type: "READ_BALANCE", value: 1000.0 },
                { seq: 5, timestamp_ns: 2262954867569, thread_name: "Race-Run01-T1", event_type: "WRITE_BALANCE", value: 1091.72 },
                { seq: 6, timestamp_ns: 2262954879598, thread_name: "Race-Run01-T1", event_type: "DEPOSIT_END", value: 91.72 },
                { seq: 7, timestamp_ns: 2262955330572, thread_name: "Race-Run01-T2", event_type: "WRITE_BALANCE", value: 908.5 },
                { seq: 8, timestamp_ns: 2262955356044, thread_name: "Race-Run01-T2", event_type: "WITHDRAW_END", value: 91.5 },
                { seq: 9, timestamp_ns: 2262955635602, thread_name: "Race-Run01-T3", event_type: "DEPOSIT_START", value: 15.29 },
                { seq: 10, timestamp_ns: 2262955644126, thread_name: "Race-Run01-T3", event_type: "READ_BALANCE", value: 908.5 },
                { seq: 11, timestamp_ns: 2262955727792, thread_name: "Race-Run01-T1", event_type: "WITHDRAW_START", value: 38.94 },
                { seq: 12, timestamp_ns: 2262955740082, thread_name: "Race-Run01-T1", event_type: "READ_BALANCE", value: 908.5 },
                { seq: 13, timestamp_ns: 2262956263325, thread_name: "Race-Run01-T4", event_type: "WITHDRAW_START", value: 74.11 },
                { seq: 14, timestamp_ns: 2262956270978, thread_name: "Race-Run01-T4", event_type: "READ_BALANCE", value: 908.5 },
                { seq: 15, timestamp_ns: 2262956416150, thread_name: "Race-Run01-T3", event_type: "WRITE_BALANCE", value: 923.79 },
                { seq: 16, timestamp_ns: 2262956427755, thread_name: "Race-Run01-T3", event_type: "DEPOSIT_END", value: 15.29 },
                { seq: 17, timestamp_ns: 2262956504760, thread_name: "Race-Run01-T2", event_type: "WITHDRAW_START", value: 87.88 },
                { seq: 18, timestamp_ns: 2262956513643, thread_name: "Race-Run01-T2", event_type: "READ_BALANCE", value: 923.79 },
                { seq: 19, timestamp_ns: 2262956581504, thread_name: "Race-Run01-T1", event_type: "WRITE_BALANCE", value: 869.56 },
                { seq: 20, timestamp_ns: 2262956590572, thread_name: "Race-Run01-T1", event_type: "WITHDRAW_END", value: 38.94 }
            ]
        },
        safe: {
            run_id: 1, variant: "safe", initial_balance: 1000.0, expected_balance: 1082.15,
            actual_balance: 1082.15, correct: true, deadlock_detected: false, truncated: true,
            events: [
                { seq: 1, timestamp_ns: 2254490537673, thread_name: "Safe-Run01-T1", event_type: "DEPOSIT_START", value: 8.83 },
                { seq: 2, timestamp_ns: 2254490561568, thread_name: "Safe-Run01-T1", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 3, timestamp_ns: 2254490568809, thread_name: "Safe-Run01-T1", event_type: "LOCK_ACQUIRED", value: null },
                { seq: 4, timestamp_ns: 2254490572799, thread_name: "Safe-Run01-T1", event_type: "READ_BALANCE", value: 1000.0 },
                { seq: 5, timestamp_ns: 2254491177479, thread_name: "Safe-Run01-T1", event_type: "WRITE_BALANCE", value: 1008.83 },
                { seq: 6, timestamp_ns: 2254491201418, thread_name: "Safe-Run01-T1", event_type: "LOCK_RELEASED", value: null },
                { seq: 7, timestamp_ns: 2254491208119, thread_name: "Safe-Run01-T1", event_type: "DEPOSIT_END", value: 8.83 },
                { seq: 8, timestamp_ns: 2254491465934, thread_name: "Safe-Run01-T2", event_type: "DEPOSIT_START", value: 32.69 },
                { seq: 9, timestamp_ns: 2254491483106, thread_name: "Safe-Run01-T2", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 10, timestamp_ns: 2254491489895, thread_name: "Safe-Run01-T2", event_type: "LOCK_ACQUIRED", value: null },
                { seq: 11, timestamp_ns: 2254491493168, thread_name: "Safe-Run01-T2", event_type: "READ_BALANCE", value: 1008.83 },
                { seq: 12, timestamp_ns: 2254492170648, thread_name: "Safe-Run01-T3", event_type: "WITHDRAW_START", value: 45.5 },
                { seq: 13, timestamp_ns: 2254492188272, thread_name: "Safe-Run01-T3", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 14, timestamp_ns: 2254492271310, thread_name: "Safe-Run01-T2", event_type: "WRITE_BALANCE", value: 1041.52 },
                { seq: 15, timestamp_ns: 2254492281527, thread_name: "Safe-Run01-T2", event_type: "LOCK_RELEASED", value: null },
                { seq: 16, timestamp_ns: 2254492325425, thread_name: "Safe-Run01-T2", event_type: "DEPOSIT_END", value: 32.69 },
                { seq: 17, timestamp_ns: 2254492428640, thread_name: "Safe-Run01-T1", event_type: "DEPOSIT_START", value: 10.01 },
                { seq: 18, timestamp_ns: 2254492441414, thread_name: "Safe-Run01-T1", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null },
                { seq: 19, timestamp_ns: 2254492522355, thread_name: "Safe-Run01-T2", event_type: "WITHDRAW_START", value: 86.1 },
                { seq: 20, timestamp_ns: 2254492544652, thread_name: "Safe-Run01-T2", event_type: "LOCK_ACQUIRE_ATTEMPT", value: null }
            ]
        }
    };

    var EVENT_COLOR = {
        DEPOSIT_START: "#5FCBE0", DEPOSIT_END: "#5FCBE0",
        WITHDRAW_START: "#5FCBE0", WITHDRAW_END: "#5FCBE0",
        READ_BALANCE: "#F0B24A", WRITE_BALANCE: "#F0B24A",
        LOCK_ACQUIRE_ATTEMPT: "#8FA79A", LOCK_ACQUIRED: "#7FE7A8", LOCK_RELEASED: "#3E7A5A",
        DEADLOCK_DETECTED: "#FF6B54", TRANSFER_COMPLETE: "#7FE7A8"
    };

    // ---------- DOM refs ----------
    var $ = function (id) { return document.getElementById(id); };
    var emptyState = $("emptyState"), results = $("results"), errMsg = $("errMsg");

    $("fileBtn").addEventListener("click", function () { $("fileInput").click(); });
    $("fileInput").addEventListener("change", function (ev) {
        var f = ev.target.files[0];
        if (!f) return;
        var reader = new FileReader();
        reader.onload = function () { $("pasteArea").value = String(reader.result); analyze(); };
        reader.readAsText(f);
    });

    // ---------- real corpus loading (trace_harness/traces/) ----------
    var CORPUS_BASE = "../trace_harness/traces";

    function runFileName(n) {
        var s = String(n);
        while (s.length < 3) s = "0" + s;
        return "trace_run_" + s + ".json";
    }

    function currentRun() {
        var v = parseInt($("runNum").value, 10);
        if (isNaN(v) || v < 1) v = 1;
        if (v > 100) v = 100;
        return v;
    }

    function setCorpusState(msg, isWarn) {
        var el = $("corpusState");
        el.textContent = msg;
        el.className = "corpus-state" + (isWarn ? " warn" : "");
    }

    Array.prototype.forEach.call(document.querySelectorAll(".sample-btn"), function (btn) {
        btn.addEventListener("click", function () {
            var variant = btn.getAttribute("data-variant");
            var run = currentRun();
            var url = CORPUS_BASE + "/" + variant + "/" + runFileName(run);

            setCorpusState("loading " + variant + " run " + run + "…", false);
            fetch(url).then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            }).then(function (trace) {
                $("pasteArea").value = JSON.stringify(trace, null, 2);
                setCorpusState("corpus · " + variant + " · run " + run, false);
                analyze();
            }).catch(function () {
                // Opened over file://, or the corpus has not been generated yet.
                $("pasteArea").value = JSON.stringify(SAMPLES[btn.getAttribute("data-sample")], null, 2);
                setCorpusState("corpus unreachable — showing bundled sample", true);
                analyze();
            });
        });
    });

    $("runBtn").addEventListener("click", analyze);

    function setLed(id, textId, on, cls, text) {
        var led = $(id);
        led.className = "led" + (on ? (" " + cls) : "");
        $(textId).textContent = text;
    }

    function analyze() {
        errMsg.textContent = "";
        var raw = $("pasteArea").value.trim();
        if (!raw) { errMsg.textContent = "Pick a sample or paste/upload a trace first."; return; }
        var trace;
        try { trace = JSON.parse(raw); }
        catch (e) { errMsg.textContent = "That isn't valid JSON: " + e.message; return; }
        if (!trace || !Array.isArray(trace.events) || trace.events.length === 0) {
            errMsg.textContent = "The trace needs a non-empty \"events\" array.";
            return;
        }

        var events = trace.events.slice().sort(function (a, b) { return a.seq - b.seq; });
        var enc = encodeRealEvents(events);
        var health = lockHealth(events);
        var flagged = Object.keys(health).filter(function (t) { return health[t].unmatched > 0; });

        renderStats(trace, events, enc, flagged);
        renderScope(enc.encodedEvents, enc.threadMap);
        renderLockTable(health, flagged);
        renderEncTable(enc, trace);
        renderLeds(trace, flagged);

        emptyState.style.display = "none";
        results.style.display = "block";
    }

    function fmtNum(n, d) { return (n === null || n === undefined) ? "&mdash;" : Number(n).toFixed(d === undefined ? 2 : d); }

    function renderStats(trace, events, enc, flagged) {
        var wallMs = ((events[events.length - 1].timestamp_ns - events[0].timestamp_ns) / 1e6);
        var threadCount = Object.keys(enc.threadMap).length;
        var cells = [
            ["RUN ID", trace.run_id !== undefined ? trace.run_id : "&mdash;", ""],
            ["VARIANT", (trace.variant || "unknown").replace("_", " "), trace.variant === "safe" ? "phosphor" : (trace.variant === "deadlock" ? "alarm" : "amber")],
            ["THREADS", threadCount, ""],
            ["EVENTS", events.length, ""],
            ["WALL TIME", wallMs.toFixed(1) + " ms", ""],
            ["INITIAL BAL.", trace.initial_balance !== undefined ? "$" + fmtNum(trace.initial_balance) : "&mdash;", ""],
            ["EXPECTED BAL.", trace.expected_balance !== undefined && trace.expected_balance !== null ? "$" + fmtNum(trace.expected_balance) : "&mdash;", ""],
            ["ACTUAL BAL.", trace.actual_balance !== undefined && trace.actual_balance !== null ? "$" + fmtNum(trace.actual_balance) : "&mdash;", ""],
            ["CORRECT", trace.correct === true ? "yes" : (trace.correct === false ? "no" : "n/a"), trace.correct === true ? "phosphor" : (trace.correct === false ? "alarm" : "")],
            ["STALLED THREADS", flagged.length, flagged.length > 0 ? "alarm" : "phosphor"]
        ];
        $("statsGrid").innerHTML = cells.map(function (c) {
            return '<div class="stat"><div class="k">' + c[0] + '</div><div class="v ' + c[2] + '">' + c[1] + '</div></div>';
        }).join("");
    }

    function renderScope(events, threadMap) {
        var svgNS = "http://www.w3.org/2000/svg";
        var svg = $("scopeSvg");
        while (svg.firstChild) svg.removeChild(svg.firstChild);

        var threads = Object.keys(threadMap).sort(function (a, b) { return threadMap[a] - threadMap[b]; });
        // Encoder indices are not lane positions: index 0 is reserved for MainThread,
        // which often has no event inside SEQ_LEN, so worker threads start at 1 and
        // the top lane would sit empty while the last thread's dots fell off the svg.
        var lanePos = {};
        threads.forEach(function (t, i) { lanePos[t] = i; });

        var laneH = 40, padL = 150, padR = 24, padT = 14, padB = 10;
        var width = Math.max(560, Math.min(1100, events.length * 16 + padL + padR));
        var height = threads.length * laneH + padT + padB;
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", height);

        var minTs = events[0].timestamp_ns, maxTs = events[events.length - 1].timestamp_ns;
        var span = Math.max(1, maxTs - minTs);
        function xOf(ts) { return padL + ((ts - minTs) / span) * (width - padL - padR); }

        threads.forEach(function (t, i) {
            var y = padT + i * laneH + laneH / 2;
            var line = document.createElementNS(svgNS, "line");
            line.setAttribute("x1", padL); line.setAttribute("x2", width - padR);
            line.setAttribute("y1", y); line.setAttribute("y2", y);
            line.setAttribute("stroke", "var(--line-soft)"); line.setAttribute("stroke-width", "1");
            svg.appendChild(line);

            var label = document.createElementNS(svgNS, "text");
            label.setAttribute("x", 4); label.setAttribute("y", y + 4);
            label.setAttribute("class", "lane-name");
            label.textContent = (t.length > 20 ? t.slice(0, 18) + "\u2026" : t);
            svg.appendChild(label);
        });

        events.forEach(function (e) {
            var y = padT + lanePos[e.thread_name] * laneH + laneH / 2;
            var x = xOf(e.timestamp_ns);
            var c = EVENT_COLOR[e.event_type] || "#8FA79A";
            var circle = document.createElementNS(svgNS, "circle");
            circle.setAttribute("cx", x); circle.setAttribute("cy", y);
            circle.setAttribute("r", e.event_type === "DEADLOCK_DETECTED" ? 6 : 4);
            circle.setAttribute("fill", c);
            circle.setAttribute("opacity", "0.92");
            var title = document.createElementNS(svgNS, "title");
            title.textContent = "#" + e.seq + " " + e.event_type + (e.value !== null && e.value !== undefined ? "  value=" + e.value : "");
            circle.appendChild(title);
            svg.appendChild(circle);
        });

        var legendItems = [
            ["#5FCBE0", "deposit / withdraw"], ["#F0B24A", "read / write balance"],
            ["#8FA79A", "lock attempt"], ["#7FE7A8", "lock acquired"],
            ["#3E7A5A", "lock released"], ["#FF6B54", "deadlock detected"]
        ];
        $("legend").innerHTML = legendItems.map(function (li) {
            return '<span class="legend-item"><i style="background:' + li[0] + '"></i>' + li[1] + '</span>';
        }).join("");
    }

    function renderLockTable(health, flagged) {
        var threads = Object.keys(health);
        var rows = threads.map(function (t) {
            var d = health[t];
            var flag = d.unmatched > 0;
            return '<tr><td>' + t + '</td><td>' + d.attempts + '</td><td>' + d.acquired + '</td><td>' + d.released + '</td>' +
                '<td>' + (flag ? '<span class="tag">stalled &middot; +' + d.unmatched + '</span>' : '<span class="tag ok">clear</span>') + '</td></tr>';
        }).join("");
        $("lockTable").innerHTML =
            '<thead><tr><th>Thread</th><th>Attempts</th><th>Acquired</th><th>Released</th><th>Status</th></tr></thead><tbody>' + rows + '</tbody>';
    }

    function renderEncTable(enc, trace) {
        var rows = enc.rows;
        var shown = rows.slice(0, 80);
        var note = "event_type_id, thread_index, value_normalized (log\u2011scaled), has_value, and delta_time_normalized for every real event";
        if (trace.truncated) {
            note += " \u2014 this is a bundled demo sample, not a corpus run";
        } else if (enc.droppedByTruncation > 0) {
            note += " \u2014 this run has " + (rows.length + enc.droppedByTruncation) +
                " events, so the encoder keeps the first " + SEQ_LEN + " and drops " +
                enc.droppedByTruncation;
        }
        $("encSub").textContent = note + " (" + shown.length + " of " + rows.length + " rows shown).";
        var head = '<thead><tr><th>seq</th><th>thread</th><th>event_type</th><th>type_id</th><th>thread_idx</th><th>value_norm</th><th>has_value</th><th>delta_norm</th></tr></thead>';
        var body = shown.map(function (r) {
            return '<tr><td class="num">' + r.seq + '</td><td>' + r.thread_name + '</td><td>' + r.event_type + '</td>' +
                '<td class="num">' + r.event_type_id + '</td><td class="num">' + r.thread_index + '</td>' +
                '<td class="num">' + r.value_normalized.toFixed(4) + '</td><td class="num">' + r.has_value + '</td>' +
                '<td class="num">' + r.delta_time_normalized.toFixed(4) + '</td></tr>';
        }).join("");
        $("encTable").innerHTML = head + "<tbody>" + body + "</tbody>";
    }

    function renderLeds(trace, flagged) {
        var vcls = trace.variant === "safe" ? "on-phosphor" : (trace.variant === "deadlock" ? "on-alarm" : "on-amber");
        setLed("ledVariant", "ledVariantText", true, vcls, trace.variant || "unknown");
        if (trace.correct === true) setLed("ledCorrect", "ledCorrectText", true, "on-phosphor", "correct");
        else if (trace.correct === false) setLed("ledCorrect", "ledCorrectText", true, "on-alarm", "corrupted");
        else setLed("ledCorrect", "ledCorrectText", false, "", "n/a");
        if (flagged.length > 0) setLed("ledDeadlock", "ledDeadlockText", true, "on-alarm", flagged.length + " thread(s) stalled");
        else setLed("ledDeadlock", "ledDeadlockText", true, "on-phosphor", "none");
    }

})();
