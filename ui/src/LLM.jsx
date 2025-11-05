import React, { useState } from "react";
import {
  Container,
  Form,
  FormControl,
  Button,
  Card,
  Row,
  Col,
} from "react-bootstrap";

const LLM = () => {
  const [inputText, setInputText] = useState("");
  const [systemText, setSystemText] = useState("");

  // Advanced options
  const [maxLength, setMaxLength] = useState("");
  const [topK, setTopK] = useState("");
  const [seed, setSeed] = useState("");
  const [device, setDevice] = useState("");
  const [modelName, setModelName] = useState("qwen3-4b");
  const [chatMode, setChatMode] = useState(false);
  const [showThinking, setShowThinking] = useState(true);

  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [showHeaders, setShowHeaders] = useState(false);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const [results, setResults] = useState("");
  const [headers, setHeaders] = useState("");

  // Combine perf-related state into a single object
  const [{ charsPerSecond }, setState] = useState({
    totalChars: 0,
    charsPerSecond: 0,
    startCharsTime: null,
  });

  // Build backend base URL from the current hostname, with fixed port 4444
  const backendBaseUrl = `http://${
    typeof window !== "undefined" ? window.location.hostname : "localhost"
  }:4444`;

  const llmPrefix = "/llm";

  const handleSubmit = async () => {
    setState((prevState) => ({
      ...prevState,
      totalChars: 0,
      charsPerSecond: 0,
      startCharsTime: null,
    }));
    setResults("");
    let collected = "";

    const encodedText = encodeURIComponent(inputText);

    const params = new URLSearchParams();
    if (maxLength !== "") params.set("max_length", maxLength);
    if (topK !== "") params.set("top_k", topK);
    if (seed !== "") params.set("seed", seed);
    if (device !== "") params.set("device", device);
    if (modelName !== "") params.set("model", modelName);
    if (chatMode) params.set("chat", "1");
    if (chatMode && systemText.trim() !== "") params.set("system", systemText);

    const url = `${backendBaseUrl}${llmPrefix}/generate/${encodedText}?${params.toString()}`;

    const response = await fetch(url, { method: "GET" });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    // Capture all headers, including any X-* headers
    const headersObj = {};
    response.headers.forEach((value, key) => {
      headersObj[key] = value;
    });
    setHeaders(JSON.stringify(headersObj, null, 2));

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const chars = chunk.length;

      setState((prevState) => {
        const newTotalChars = prevState.totalChars + chars;
        let startTime = prevState.startCharsTime;
        if (startTime === null) {
          startTime = performance.now();
        }
        const currentTime = performance.now();
        const timeElapsed = (currentTime - startTime) / 1000; // seconds
        const newCharsPerSecond = newTotalChars / Math.max(timeElapsed, 1e-9);

        return {
          totalChars: newTotalChars,
          charsPerSecond: newCharsPerSecond,
          startCharsTime: startTime,
        };
      });

      collected += chunk;
      setResults((prevResults) => prevResults + chunk);
    }

    // After complete stream, optionally strip thinking blocks
    if (!showThinking) {
      const withoutThinking = collected.replace(/<think>[\s\S]*?<\/think>/g, "");
      setResults(withoutThinking);
    }
  };

  return (
    <>
      <Container className="llm-container mt-3">
        {!chatMode && (
          <Row className="mb-3">
            <Col>
              <FormControl
                type="text"
                placeholder='Enter text, e.g. "The quick brown fox jumps over"'
                value={inputText}
                onChange={handleInputChange}
              />
            </Col>
            <Col xs="auto">
              <Button
                variant="primary"
                onClick={handleSubmit}
                disabled={!inputText.trim()}
              >
                Submit
              </Button>
            </Col>
          </Row>
        )}

        {chatMode && (
          <>
            <Row className="mb-2">
              <Col>
                <Form.Label className="mb-1">System Prompt</Form.Label>
                <FormControl
                  as="textarea"
                  rows={3}
                  placeholder="Optional system instructions for the assistant"
                  value={systemText}
                  onChange={(e) => setSystemText(e.target.value)}
                />
              </Col>
            </Row>
            <Row className="mb-3">
              <Col>
                <Form.Label className="mb-1">User Prompt</Form.Label>
                <FormControl
                  as="textarea"
                  rows={6}
                  placeholder="Enter user prompt/content"
                  value={inputText}
                  onChange={handleInputChange}
                />
              </Col>
            </Row>
            <Row className="mb-3">
              <Col xs="auto">
                <Button
                  variant="primary"
                  onClick={handleSubmit}
                  disabled={!inputText.trim()}
                >
                  Submit
                </Button>
              </Col>
            </Row>
          </>
        )}
        <Row className="mb-3">
          <Col>
            <Form.Check
              type="switch"
              id="advanced-options-toggle-llm"
              label="Show Advanced Options"
              checked={showAdvancedOptions}
              onChange={() => setShowAdvancedOptions(!showAdvancedOptions)}
            />
          </Col>
          <Col>
            <Form.Check
              type="switch"
              id="show-headers-toggle-llm"
              label="Show Headers"
              checked={showHeaders}
              onChange={() => setShowHeaders(!showHeaders)}
            />
          </Col>
        </Row>
      </Container>
      <Container>
        {showAdvancedOptions && (
          <div className="p-3 border rounded">
            <Form className="d-flex flex-wrap mb-3">
              <Form.Group className="d-flex align-items-center me-2 mb-2">
                <Form.Label className="me-2 mb-0" style={{ whiteSpace: "nowrap" }}>
                  Max Length
                </Form.Label>
                <FormControl
                  type="number"
                  placeholder="Max Length"
                  value={maxLength}
                  onChange={(e) => setMaxLength(e.target.value)}
                  min="1"
                  max="1000000"
                />
              </Form.Group>

              <Form.Group className="d-flex align-items-center me-2 mb-2">
                <Form.Label className="me-2 mb-0" style={{ whiteSpace: "nowrap" }}>
                  Top-K
                </Form.Label>
                <FormControl
                  type="number"
                  placeholder="Top-K"
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  min="1"
                  max="100000"
                />
              </Form.Group>

              <Form.Group className="d-flex align-items-center me-2 mb-2">
                <Form.Label className="me-2 mb-0">Seed</Form.Label>
                <FormControl
                  type="number"
                  placeholder="Seed"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
              </Form.Group>

              <Form.Group className="d-flex align-items-center me-2 mb-2">
                <Form.Label className="me-2 mb-0">Device</Form.Label>
                <Form.Select value={device} onChange={(e) => setDevice(e.target.value)}>
                  <option value="">Select a device</option>
                  <option value="cuda">cuda</option>
                  <option value="cuda:0">cuda:0</option>
                  <option value="cuda:1">cuda:1</option>
                  <option value="cuda:2">cuda:2</option>
                  <option value="cuda:3">cuda:3</option>
                  <option value="cpu">cpu</option>
                </Form.Select>
              </Form.Group>

              <Form.Group className="d-flex align-items-center me-2 mb-2">
                <Form.Label className="me-2 mb-0" style={{ whiteSpace: "nowrap" }}>
                  Model
                </Form.Label>
                <Form.Select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                  <option value="qwen3-4b">qwen3-4b</option>
                  <option value="gpt-oss-20b">gpt-oss-20b</option>
                  <option value="gpt-oss-120b">gpt-oss-120b</option>
                  <option value="lfm2-8b-a1b">lfm2-8b-a1b</option>
                  <option value="llama2-7b">llama2-7b</option>
                </Form.Select>
              </Form.Group>

              <Form.Group className="d-flex align-items-center me-2 mb-2">
                <Form.Check
                  type="switch"
                  id="chat-template-toggle"
                  label="Chat Template"
                  checked={chatMode}
                  onChange={() => setChatMode(!chatMode)}
                />
              </Form.Group>
            </Form>
            <Form className="d-flex flex-wrap mb-2">
              <Form.Group className="d-flex align-items-center me-3 mb-2">
                <Form.Label className="me-2 mb-0" style={{ whiteSpace: "nowrap" }}>
                  Show Thinking
                </Form.Label>
                <Form.Check
                  type="radio"
                  id="show-thinking-yes"
                  name="show-thinking"
                  label="Yes"
                  className="me-3"
                  checked={showThinking === true}
                  onChange={() => setShowThinking(true)}
                />
                <Form.Check
                  type="radio"
                  id="show-thinking-no"
                  name="show-thinking"
                  label="No"
                  checked={showThinking === false}
                  onChange={() => setShowThinking(false)}
                />
              </Form.Group>
            </Form>
          </div>
        )}

        <Card className="mt-3">
          {showHeaders && (
            <Card.Body className="headers-area">
              <pre>{headers}</pre>
            </Card.Body>
          )}
          <Card.Body className="results-area">
            {results || "Results will be displayed here."}
          </Card.Body>
          <Card.Footer className="text-muted">
            {charsPerSecond.toFixed(2)} chars/s {" "}
            {charsPerSecond < 3
              ? "(< 1 tok/s)"
              : `(~${Math.round(charsPerSecond / 3)} tok/s)`}
          </Card.Footer>
        </Card>
      </Container>
    </>
  );
};

export default LLM;


