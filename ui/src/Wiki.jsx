import React, { useState, useRef, useEffect } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { solarizedlight } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Container,
  Form,
  FormControl,
  ListGroup,
  Card,
} from "react-bootstrap";
import { bytesToHuman } from "./Utils";

const Wiki = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [selectedXml, setSelectedXml] = useState(null);
  const [format, setFormat] = useState("Raw");
  const [shouldSearch, setShouldSearch] = useState(true);
  const [selectedHtml, setSelectedHtml] = useState(null);
  const [searchStatus, setSearchStatus] = useState("");
  const [selectedRange, setSelectedRange] = useState(null); // { name, startByte, endByte }

  // Build backend base URL from the current hostname, with fixed port 4444
  const backendBaseUrl = `http://${
    typeof window !== "undefined" ? window.location.hostname : "localhost"
  }:4444`;

  const wikiPrefix = "/wiki";

  // Decode HTML entities (including common ones missing the semicolon like &amp or &quot)
  const normalizeHtmlEntities = (str) =>
    typeof str === "string"
      ? str.replace(/&(amp|quot|apos|lt|gt)(?!;)/g, "&$1;")
      : str;

  const decodeHtmlEntities = (str) => {
    if (typeof window === "undefined" || typeof document === "undefined") {
      return str;
    }
    const normalized = normalizeHtmlEntities(str);
    const textarea = document.createElement("textarea");
    textarea.innerHTML = normalized;
    return textarea.value;
  };

  // Debounce search function and abort controller for cancelling requests
  const abortControllerRef = useRef(null);
  const handleSearch = (e) => {
    setQuery(e.target.value);
    setShouldSearch(true);
    setSelectedXml(null); // Clear selected content when the search box is cleared
    setSelectedHtml(null);
    setSelectedRange(null);
  };

  useEffect(() => {
    if (!shouldSearch || query.trim().length < 3) {
      setResults([]);
      //setSearchStatus("");
      return;
    }

    setSearchStatus(`Searching for '${query}'...`);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort(); // Cancel any ongoing fetch requests
    }

    const startTime = performance.now();
    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    const fetchData = async () => {
      try {
        const response = await fetch(
          `${backendBaseUrl}${wikiPrefix}/offsets/${encodeURIComponent(query)}`,
          {
          mode: "cors",
          signal,
          }
        );
        const data = await response.json();
        setResults(data);
        const endTime = performance.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        setSearchStatus(
          `Received ${data.length} results for '${query}' in ${duration} seconds.`
        );
      } catch (error) {
        if (error.name !== "AbortError") {
          console.error("Error fetching search results:", error);
          setResults([]);
        }
      }
    };

    const timeoutId = setTimeout(fetchData, 1000);

    return () => clearTimeout(timeoutId);
  }, [query, shouldSearch, backendBaseUrl]);

  // Fetch content for a given byte range using current format
  const fetchContent = async (name, startByte, endByte) => {
    const startTime = performance.now();
    try {
      const url =
        format === "Raw"
          ? `${backendBaseUrl}${wikiPrefix}/xml`
          : `${backendBaseUrl}${wikiPrefix}/html`;
      const response = await fetch(url, {
        headers: {
          Range: `bytes=${startByte}-${endByte}`,
        },
      });
      const data = await response.text();
      const contentLength = response.headers.get("Content-Length");
      const endTime = performance.now();
      const duration = ((endTime - startTime) / 1000).toFixed(2);
      const msg = `Received ${bytesToHuman(contentLength)} in ${duration} seconds.`;
      console.log(msg);
      setShouldSearch(false);
      setQuery(decodeHtmlEntities(name)); // Place the decoded result name into the search bar
      if (format === "Raw") {
        setSelectedXml(data);
        setSelectedHtml(null);
      } else {
        setSelectedHtml(data);
        setSelectedXml(null);
      }
      setSearchStatus(msg);
      setResults([]); // Clear results when an item is clicked
    } catch (error) {
      console.error("Error fetching content:", error);
      if (format === "Raw") {
        setSelectedXml(null);
      } else {
        setSelectedHtml(null);
      }
    }
  };

  // Handle item click and fetch data based on selected format
  const handleResultClick = async (name, startByte, endByte) => {
    setSelectedRange({ name, startByte, endByte });
    fetchContent(name, startByte, endByte);
  };

  // When format changes and a selection exists, re-fetch same range with new format
  useEffect(() => {
    if (!selectedRange) return;
    const { name, startByte, endByte } = selectedRange;
    fetchContent(name, startByte, endByte);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [format]);

  return (
    <Container className="wiki-search-container">
      <Form className="mb-3">
        <FormControl
          type="search"
          placeholder="Search"
          className="me-2"
          aria-label="Search"
          value={query}
          onChange={handleSearch}
        />
        <div className="mt-2">
          <Form.Check
            inline
            label="Raw"
            name="render-mode"
            type="radio"
            id="render-mode-raw"
            checked={format === "Raw"}
            onChange={() => setFormat("Raw")}
          />
          <Form.Check
            inline
            label="Pandoc"
            name="render-mode"
            type="radio"
            id="render-mode-pandoc"
            checked={format === "Pandoc"}
            onChange={() => setFormat("Pandoc")}
          />
        </div>
      </Form>

      {query && searchStatus && (
        <div className="search-status mt-2 text-muted">{searchStatus}</div>
      )}
      {/* Only render the ListGroup when results are available. */}
      {results.length > 0 && (
        <ListGroup className="mt-3">
          {results.map(([name, startByte, endByte]) => {
            const size = endByte - startByte;
            const displayName = decodeHtmlEntities(name);
            return (
              <ListGroup.Item
                key={`${name}-${startByte}`}
                action
                onClick={() => handleResultClick(name, startByte, endByte)}
              >
                {displayName} [{bytesToHuman(size)}]
              </ListGroup.Item>
            );
          })}
        </ListGroup>
      )}
      {selectedXml && (
        <Card className="mt-3">
          <Card.Body>
            <SyntaxHighlighter
              language="xml"
              style={solarizedlight}
              wrapLongLines={true}
              wrapLines={true}
            >
              {selectedXml}
            </SyntaxHighlighter>
          </Card.Body>
        </Card>
      )}
      {selectedHtml && (
        <Card className="mt-3">
          <Card.Body>
            <div dangerouslySetInnerHTML={{ __html: selectedHtml }} />
          </Card.Body>
        </Card>
      )}
    </Container>
  );
};

export default Wiki;
