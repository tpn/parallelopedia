import React, { useEffect, useRef, useState } from "react";
import {
  Container,
  Form,
  FormControl,
  Button,
  Card,
  Row,
  Col,
} from "react-bootstrap";

const GPT2 = () => {
  const [inputText, setInputText] = useState("");

  const [maxLength, setMaxLength] = useState("");
  const [seed, setSeed] = useState("");
  const [device, setDevice] = useState("");

  const [modelName, setModelName] = useState("gpt2");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [showHeaders, setShowHeaders] = useState(false);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const [results, setResults] = useState("");
  const [headers, setHeaders] = useState("");

  const inputRef = useRef(null);
  const footerRef = useRef(null);

  const DEFAULT_PHRASE = "Albert Einstein's Theory of Relativity stated that";

  // Global keybinding: press 'd' (when not typing in an input) to prefill the input
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.defaultPrevented) return;
      const target = e.target;
      const tag = target && target.tagName ? target.tagName.toUpperCase() : "";
      const isEditable =
        (target && target.isContentEditable) ||
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT";
      if (isEditable) return;

      if (
        e.key === "d" &&
        !e.ctrlKey &&
        !e.metaKey &&
        !e.altKey &&
        !e.shiftKey
      ) {
        e.preventDefault();
        setInputText(DEFAULT_PHRASE);
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Combine all related state into a single object
  const [{ charsPerSecond }, setState] = useState({
    totalChars: 0,
    charsPerSecond: 0,
    startCharsTime: null,
  });

  // Build backend base URL from the current hostname, with fixed port 4444
  const backendBaseUrl = `http://${
    typeof window !== "undefined" ? window.location.hostname : "localhost"
  }:4444`;

  const gpt2Prefix = "/gpt2";

  // Auto-scroll to keep the footer (rate) visible while streaming
  useEffect(() => {
    if (footerRef.current) {
      footerRef.current.scrollIntoView({ behavior: "auto", block: "end" });
    }
  }, [results, charsPerSecond]);

  const handleSubmit = async () => {
    setState((prevState) => ({
      ...prevState,
      totalChars: 0, // Reset total characters
      charsPerSecond: 0, // Reset characters per second
      startCharsTime: null, // Reset start character time
    }));
    setResults(""); // Clear previous results

    const textToUse = inputText.trim().length === 0 ? DEFAULT_PHRASE : inputText;
    if (textToUse !== inputText) {
      setInputText(textToUse);
    }

    const encodedText = encodeURIComponent(textToUse);
    const response = await fetch(
      `${backendBaseUrl}${gpt2Prefix}/generate/${encodedText}?max_length=${maxLength}&seed=${seed}&device=${device}&model=${modelName}`,
      {
        method: "GET",
      }
    );

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    // Capture all headers, including any X-GPU-* headers
    const headersObj = {};
    response.headers.forEach((value, key) => {
      headersObj[key] = value;
    });
    setHeaders(JSON.stringify(headersObj, null, 2));

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const chars = chunk.length; // Calculate character length of the chunk

      setState((prevState) => {
        const newTotalChars = prevState.totalChars + chars;
        let startTime = prevState.startCharsTime;
        if (startTime === null) {
          startTime = performance.now();
        }
        const currentTime = performance.now();
        const timeElapsed = (currentTime - startTime) / 1000; // Convert to seconds
        const newCharsPerSecond = newTotalChars / timeElapsed;

        return {
          totalChars: newTotalChars,
          charsPerSecond: newCharsPerSecond,
          startCharsTime: startTime,
        };
      });

      setResults((prevResults) => prevResults + chunk);
    }
  };

  const handleMaxLengthChange = (e) => {
    setMaxLength(e.target.value);
  };

  const handleSeedChange = (e) => {
    setSeed(e.target.value);
  };

  const handleDeviceChange = (e) => {
    setDevice(e.target.value);
  };

  const handleModelNameChange = (e) => {
    setModelName(e.target.value);
  };

  return (
    <>
      <Container className="llm-container mt-3">
        <Row className="mb-3">
          <Col>
            <FormControl
              type="text"
              placeholder="Enter text, e.g. &#34;Albert Einstein&#39;s Theory of Relativity stated that&#34;"
              value={inputText}
              onChange={handleInputChange}
            ref={inputRef}
            />
          </Col>
          <Col xs="auto">
            <Button
              variant="primary"
            onClick={handleSubmit}
            >
              Submit
            </Button>
          </Col>
        </Row>
        <Row className="mb-3">
          <Col>
            <Form.Check
              type="switch"
              id="advanced-options-toggle"
              label="Show Advanced Options"
              checked={showAdvancedOptions}
              onChange={() => setShowAdvancedOptions(!showAdvancedOptions)}
            />
          </Col>
          <Col>
            <Form.Check
              type="switch"
              id="show-headers-toggle"
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
            <Form className="d-flex mb-3">
              <Form.Group className="d-flex align-items-center me-2">
                <Form.Label
                  className="me-2 mb-0"
                  style={{ whiteSpace: "nowrap" }}
                >
                  Max Length
                </Form.Label>
                <FormControl
                  type="number"
                  placeholder="Max Length"
                  value={maxLength}
                  onChange={handleMaxLengthChange}
                  min="10"
                  max="1024"
                />
              </Form.Group>
              <Form.Group className="d-flex align-items-center me-2">
                <Form.Label className="me-2 mb-0">Seed</Form.Label>
                <FormControl
                  type="number"
                  placeholder="Seed"
                  value={seed}
                  onChange={handleSeedChange}
                />
              </Form.Group>
              <Form.Group className="d-flex align-items-center me-2">
                <Form.Label className="me-2 mb-0">Device</Form.Label>
                <Form.Select value={device} onChange={handleDeviceChange}>
                  <option value="">Select a device</option>
                  <option value="cuda">cuda</option>
                  <option value="cuda:0">cuda:0</option>
                  <option value="cuda:1">cuda:1</option>
                  <option value="cuda:2">cuda:2</option>
                  <option value="cuda:3">cuda:3</option>
                  <option value="cpu">cpu</option>
                </Form.Select>
              </Form.Group>
              <Form.Group className="d-flex align-items-center me-2">
                <Form.Label
                  className="me-2 mb-0"
                  style={{ whiteSpace: "nowrap" }}
                >
                  Model Name
                </Form.Label>
                <Form.Select value={modelName} onChange={handleModelNameChange}>
                  <option value="gpt2">gpt2</option>
                  <option value="gpt2-xl">gpt2-xl</option>
                </Form.Select>
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
          <Card.Footer className="text-muted" ref={footerRef}>
            {charsPerSecond.toFixed(2)} chars/s{" "}
            {charsPerSecond < 3
              ? "(< 1 tok/s)"
              : `(~${Math.round(charsPerSecond / 3)} tok/s)`}
          </Card.Footer>
        </Card>
      </Container>
    </>
  );
};

export default GPT2;
