//! Integration tests that drive the built `metalroute` binary end-to-end.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_metalroute");

const SAMPLE: &str = r#"{
    "layerCount": 2,
    "bounds": { "minX": 0, "maxX": 10, "minY": 0, "maxY": 10 },
    "obstacles": [
        { "type": "rect", "center": {"x": 5, "y": 5}, "width": 2, "height": 2 }
    ],
    "connections": [
        { "name": "VCC", "pointsToConnect": [ {"x": 1, "y": 1}, {"x": 9, "y": 1} ] },
        { "name": "GND", "pointsToConnect": [ {"x": 1, "y": 9}, {"x": 9, "y": 9} ] }
    ]
}"#;

/// Write the sample SRJ to a unique temp file and return its path.
fn write_sample() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!(
        "mr_cli_test_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    path.push(unique);
    std::fs::write(&path, SAMPLE).unwrap();
    path
}

#[test]
fn route_to_stdout_emits_pcb_traces_and_summary() {
    let input = write_sample();

    let output = Command::new(BIN)
        .args(["route", "--input"])
        .arg(&input)
        .args(["--resolution", "1.0", "--router", "ripup"])
        .output()
        .expect("failed to run metalroute");

    let _ = std::fs::remove_file(&input);

    assert!(
        output.status.success(),
        "exit: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let traces: serde_json::Value = serde_json::from_str(&stdout).expect("solution must be JSON");
    let arr = traces.as_array().expect("solution must be a JSON array");
    assert!(!arr.is_empty(), "expected >=1 pcb_trace");
    assert!(
        arr.iter().filter(|t| t["type"] == "pcb_trace").count() >= 1,
        "expected at least one pcb_trace element"
    );

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("2/2"),
        "summary should report 2/2 nets, got: {stderr}"
    );
    // Non-uniform / Hanan grid (Phase 3): bounds + pad endpoints + obstacle edges +
    // fill channels. route_problem now applies the default copper clearance, so a fill
    // channel must fit `track_w + 2·clearance` rather than a bare track — fewer midpoint
    // lanes are inserted than the old no-clearance build, yielding 10 lines per axis.
    assert!(stderr.contains("10x10"), "summary should report grid dims");
}

#[test]
fn route_to_file_writes_solution() {
    let input = write_sample();
    let mut out = std::env::temp_dir();
    out.push(format!(
        "mr_cli_out_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let output = Command::new(BIN)
        .args(["route", "--input"])
        .arg(&input)
        .args(["--out"])
        .arg(&out)
        .output()
        .expect("failed to run metalroute");

    let _ = std::fs::remove_file(&input);

    assert!(output.status.success());
    let written = std::fs::read_to_string(&out).expect("output file must exist");
    let _ = std::fs::remove_file(&out);
    let traces: serde_json::Value = serde_json::from_str(&written).unwrap();
    assert!(traces.as_array().is_some());
}

#[test]
fn project_large_batch_reports_go() {
    let output = Command::new(BIN)
        .args([
            "project", "--width", "256", "--height", "256", "--nets", "500",
        ])
        .output()
        .expect("failed to run metalroute");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("GO"), "stdout: {stdout}");
    assert!(!stdout.contains("NO-GO"), "stdout: {stdout}");
}

#[test]
fn project_tiny_reports_no_go() {
    let output = Command::new(BIN)
        .args(["project", "--width", "8", "--height", "8", "--nets", "1"])
        .output()
        .expect("failed to run metalroute");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("NO-GO"), "stdout: {stdout}");
}

#[test]
fn help_smoke_test() {
    let output = Command::new(BIN)
        .arg("--help")
        .output()
        .expect("failed to run metalroute");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("route"),
        "help should list route subcommand"
    );
    assert!(
        stdout.contains("project"),
        "help should list project subcommand"
    );
}

#[test]
fn missing_input_file_errors() {
    let output = Command::new(BIN)
        .args(["route", "--input", "/no/such/file/definitely.json"])
        .output()
        .expect("failed to run metalroute");
    assert!(!output.status.success(), "missing input should fail");
}
