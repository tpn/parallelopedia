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
  const [format] = useState("XML");
  const [shouldSearch, setShouldSearch] = useState(true);
  const [selectedHtml, setSelectedHtml] = useState(null);
  const [searchStatus, setSearchStatus] = useState("");

  // Build backend base URL from the current hostname, with fixed port 4444
  const backendBaseUrl = `http://${
    typeof window !== "undefined" ? window.location.hostname : "localhost"
  }:4444`;

  const wikiPrefix = "/wiki";

  // Debounce search function and abort controller for cancelling requests
  const abortControllerRef = useRef(null);
  const handleSearch = (e) => {
    setQuery(e.target.value);
    setShouldSearch(true);
    setSelectedXml(null); // Clear selected XML when the search box is cleared
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

  // Handle item click and fetch data based on selected format
  const handleResultClick = async (name, startByte, endByte) => {
    const startTime = performance.now();
    try {
      const url =
        format === "XML"
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
      setShouldSearch(false);
      setQuery(name); // Place the result's name into the search bar
      if (format === "XML") {
        setSelectedXml(data);
        setSelectedHtml(null);
        var msg = `Received ${bytesToHuman(
          contentLength
        )} in ${duration} seconds.`;
        console.log(msg);
        setSearchStatus(msg);
      } else {
        setSelectedHtml(data);
        setSelectedXml(null);
      }
      setResults([]); // Clear results when an item is clicked
    } catch (error) {
      console.error("Error fetching XML data:", error);
      if (format === "XML") {
        setSelectedXml(null);
      } else {
        setSelectedHtml(null);
      }
    }
  };

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
      </Form>

      {query && searchStatus && (
        <div className="search-status mt-2 text-muted">{searchStatus}</div>
      )}
      {/* Only render the ListGroup when results are available. */}
      {results.length > 0 && (
        <ListGroup className="mt-3">
          {results.map(([name, startByte, endByte]) => {
            const size = endByte - startByte;
            return (
              <ListGroup.Item
                key={`${name}-${startByte}`}
                action
                onClick={() => handleResultClick(name, startByte, endByte)}
              >
                {name} [{bytesToHuman(size)}]
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
