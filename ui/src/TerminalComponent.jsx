import React, { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";

const TerminalComponent = ({ output }) => {
  const terminalRef = useRef(null);
  const xtermRef = useRef(null);
  const fitAddon = useRef(null);

  useEffect(() => {
    // Initialize xterm.js
    xtermRef.current = new Terminal({
      cursorBlink: true,
      rows: 80,
      cols: 120,
      theme: {
        background: "#1e1e1e",
        foreground: "#ffffff",
      },
      fontFamily: "Comic Mono NF",
      fontSize: 14,
    });

    fitAddon.current = new FitAddon();
    xtermRef.current.loadAddon(fitAddon.current);

    xtermRef.current.open(terminalRef.current);
    fitAddon.current.fit();

    return () => {
      xtermRef.current.dispose();
    };
  }, []);

  useEffect(() => {
    // Append new output to the terminal
    if (xtermRef.current) {
      xtermRef.current.write(output);
    }
  }, [output]);

  return (
    <div
      ref={terminalRef}
      style={{
        width: "810px",
        height: "800px",
        border: "1px solid #ccc",
        fontFamily: "Comic Mono NF, monospace",
        fontSize: "14px",
      }}
    />
  );
};

export default TerminalComponent;
