import React, { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";

const TerminalTab = () => {
  const terminalRef = useRef(null);
  const xtermRef = useRef(null);
  const fitAddon = useRef(null);

  useEffect(() => {
    // Initialize xterm.js
    xtermRef.current = new Terminal({
      cursorBlink: true,
      theme: {
        background: "#1e1e1e",
        foreground: "#ffffff",
      },
    });

    fitAddon.current = new FitAddon();
    xtermRef.current.loadAddon(fitAddon.current);

    // Open the terminal in the div
    xtermRef.current.open(terminalRef.current);

    const fitSafely = () => {
      if (fitAddon.current) {
        try {
          fitAddon.current.fit();
        } catch (e) {
          // swallow transient fit timing errors; we'll retry on observers
        }
      }
    };

    // Fit after next paint and when fonts are ready
    if (typeof window !== "undefined" && window.requestAnimationFrame) {
      window.requestAnimationFrame(() => fitSafely());
    } else {
      setTimeout(fitSafely, 0);
    }
    if (document && document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => fitSafely());
    }

    // Write an initial message
    //xtermRef.current.write("Hello from xterm.js\r\n");

    // WebSocket connection
    const protocol =
      typeof window !== "undefined" && window.location.protocol === "https:"
        ? "wss"
        : "ws";
    const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
    const ws = new WebSocket(`${protocol}://${host}:9091`);
    ws.onopen = () => {
      //xtermRef.current.write("Connected to the local terminal\r\n");
    };

    ws.onmessage = (event) => {
      xtermRef.current.write(event.data);
    };

    xtermRef.current.onData((data) => {
      ws.send(data); // Send user input to the backend shell
    });

    // Handle window resize and container resize
    const handleResize = () => fitSafely();
    window.addEventListener("resize", handleResize);
    let ro = null;
    if (typeof ResizeObserver !== "undefined" && terminalRef.current) {
      ro = new ResizeObserver(() => fitSafely());
      ro.observe(terminalRef.current);
    }

    return () => {
      xtermRef.current.dispose();
      ws.close();
      window.removeEventListener("resize", handleResize);
      if (ro) {
        try { ro.disconnect(); } catch (e) {}
      }
    };
  }, []);

  return (
    <div
      ref={terminalRef}
      style={{ width: "100%", height: "810px", border: "1px solid #ccc" }}
    />
  );
};

export default TerminalTab;
