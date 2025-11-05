import React, { useEffect, useMemo, useState } from "react";
import Stats from "./Stats";
import Container from "react-bootstrap/Container";
import Wiki from "./Wiki";
import GPT2 from "./GPT2";
import LLM from "./LLM";
import BtopTab from "./BtopTab";
import TerminalTab from "./TerminalTab";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.scss";
import "./App.css";
import { Tabs } from "react-bootstrap";
import Tab from "react-bootstrap/Tab";
import Modal from "react-bootstrap/Modal";
import Form from "react-bootstrap/Form";

const App = () => {
  const [showSettings, setShowSettings] = useState(false);
  const [visibleApps, setVisibleApps] = useState({ wiki: true, gpt2: true, llm: true, btop: false, terminal: false });
  const [activeKey, setActiveKey] = useState("wiki");

  const availableApps = useMemo(
    () => [
      { id: "wiki", label: "Wiki" },
      { id: "gpt2", label: "GPT2" },
      { id: "llm", label: "LLM" },
      { id: "btop", label: "Btop" },
      { id: "terminal", label: "Terminal" },
    ],
    []
  );

  const tabOrder = useMemo(() => ["wiki", "gpt2", "llm", "btop", "terminal"], []);

  const accelerators = useMemo(() => {
    const used = new Set();
    const map = {};
    for (const app of availableApps) {
      const label = app.label;
      let chosen = { key: null, index: -1 };
      for (let i = 0; i < label.length; i++) {
        const ch = label[i];
        if (!/[a-z]/i.test(ch)) continue;
        const lower = ch.toLowerCase();
        if (!used.has(lower)) {
          used.add(lower);
          chosen = { key: lower, index: i };
          break;
        }
      }
      if (chosen.index === -1) {
        // Fallback to first character if nothing suitable was found
        const idx = Math.max(0, label.split("").findIndex((c) => /[a-z]/i.test(c)));
        const lower = (label[idx] || app.id[0]).toLowerCase();
        chosen = { key: lower, index: idx };
      }
      map[app.id] = chosen;
    }
    return map;
  }, [availableApps]);

  const renderWithUnderline = (label, accelIndex, accelKey) => {
    if (accelIndex < 0 || accelIndex >= label.length) return label;
    return (
      <span aria-keyshortcuts={accelKey}>
        {label.slice(0, accelIndex)}
        <u>{label[accelIndex]}</u>
        {label.slice(accelIndex + 1)}
      </span>
    );
  };

  useEffect(() => {
    if (!visibleApps[activeKey]) {
      const nextKey = tabOrder.find((k) => visibleApps[k]);
      if (nextKey) {
        setActiveKey(nextKey);
      }
    }
  }, [visibleApps, activeKey, tabOrder]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "Escape" || e.key === "Esc") {
        if (!showSettings) {
          setShowSettings(true);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showSettings]);

  useEffect(() => {
    if (!showSettings) return;
    const onKeyDown = (e) => {
      // Ignore when typing in inputs/textareas/contenteditable
      const tag = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || e.target.isContentEditable) return;

      // Close on Enter
      if (e.key === "Enter") {
        e.preventDefault();
        setShowSettings(false);
        return;
      }

      // Ensure Escape closes when modal is open
      if (e.key === "Escape" || e.key === "Esc") {
        e.preventDefault();
        setShowSettings(false);
        return;
      }

      if (e.key && e.key.length === 1) {
        const key = e.key.toLowerCase();
        const match = availableApps.find((app) => accelerators[app.id]?.key === key);
        if (match) {
          e.preventDefault();
          setVisibleApps((v) => ({ ...v, [match.id]: !v[match.id] }));
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showSettings, availableApps, accelerators]);

  return (
    <Container>
      <header className="py-3 app-header">
        <h1 className="hero-title10">Parallelopedia</h1>
        <button
          className="settings-button"
          aria-label="Settings"
          title="Settings"
          onClick={() => setShowSettings(true)}
        >
          ⚙️
        </button>
      </header>
      <main className="mt-3">
        <Tabs activeKey={activeKey} onSelect={(k) => k && setActiveKey(k)} id="main-tabs" mountOnEnter unmountOnExit>
          {visibleApps.wiki && (
            <Tab title="Wiki" eventKey="wiki">
              <Wiki />
            </Tab>
          )}
          {false && (
            <Tab title="Stats" eventKey="stats">
              <Stats />
            </Tab>
          )}
          {visibleApps.gpt2 && (
            <Tab title="GPT2" eventKey="gpt2">
              <GPT2 />
            </Tab>
          )}
          {visibleApps.llm && (
            <Tab title="LLM" eventKey="llm">
              <LLM />
            </Tab>
          )}
          {visibleApps.btop && (
            <Tab title="Btop" eventKey="btop">
              <BtopTab />
            </Tab>
          )}
          {visibleApps.terminal && (
            <Tab title="Terminal" eventKey="terminal">
              <TerminalTab />
            </Tab>
          )}
        </Tabs>
      </main>

      <Modal show={showSettings} onHide={() => setShowSettings(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Settings</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Check
              type="checkbox"
              id="toggle-wiki"
              label={renderWithUnderline(
                "Wiki",
                accelerators.wiki?.index ?? 0,
                accelerators.wiki?.key
              )}
              checked={visibleApps.wiki}
              onChange={(e) => setVisibleApps((v) => ({ ...v, wiki: e.target.checked }))}
            />
            <Form.Check
              className="mt-2"
              type="checkbox"
              id="toggle-gpt2"
              label={renderWithUnderline(
                "GPT2",
                accelerators.gpt2?.index ?? 0,
                accelerators.gpt2?.key
              )}
              checked={visibleApps.gpt2}
              onChange={(e) => setVisibleApps((v) => ({ ...v, gpt2: e.target.checked }))}
            />
            <Form.Check
              className="mt-2"
              type="checkbox"
              id="toggle-llm"
              label={renderWithUnderline(
                "LLM",
                accelerators.llm?.index ?? 0,
                accelerators.llm?.key
              )}
              checked={visibleApps.llm}
              onChange={(e) => setVisibleApps((v) => ({ ...v, llm: e.target.checked }))}
            />
            <Form.Check
              className="mt-2"
              type="checkbox"
              id="toggle-btop"
              label={renderWithUnderline(
                "Btop",
                accelerators.btop?.index ?? 0,
                accelerators.btop?.key
              )}
              checked={visibleApps.btop}
              onChange={(e) => setVisibleApps((v) => ({ ...v, btop: e.target.checked }))}
            />
            <Form.Check
              className="mt-2"
              type="checkbox"
              id="toggle-terminal"
              label={renderWithUnderline(
                "Terminal",
                accelerators.terminal?.index ?? 0,
                accelerators.terminal?.key
              )}
              checked={visibleApps.terminal}
              onChange={(e) => setVisibleApps((v) => ({ ...v, terminal: e.target.checked }))}
            />
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <button className="btn btn-secondary" onClick={() => setShowSettings(false)}>
            Close
          </button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default App;
