from tui.runner import stream_subprocess, RunnerOutput, RunnerDone

def test_stream_yields_lines_then_exit_zero():
    results = list(stream_subprocess(["python3", "-c", "print('a'); print('b')"]))
    lines = [line for line, code in results if line is not None]
    finals = [code for line, code in results if code is not None]
    assert lines == ["a", "b"]
    assert finals == [0]

def test_stream_nonzero_exit():
    results = list(stream_subprocess(["python3", "-c", "import sys; sys.exit(2)"]))
    finals = [code for line, code in results if code is not None]
    assert finals == [2]

def test_stream_stderr_merged():
    results = list(stream_subprocess(
        ["python3", "-c", "import sys; sys.stderr.write('err\\n'); print('out')"]
    ))
    lines = [line for line, _ in results if line is not None]
    assert "err" in lines and "out" in lines

def test_runner_messages_are_textual_messages():
    from textual.message import Message
    assert issubclass(RunnerOutput, Message)
    assert issubclass(RunnerDone, Message)
    assert RunnerOutput("hi").line == "hi"
    assert RunnerDone(42).exit_code == 42
