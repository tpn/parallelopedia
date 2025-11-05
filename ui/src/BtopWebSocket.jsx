import React, { useEffect, useState } from "react";

const BtopWebSocket = () => {
  const [output, setOutput] = useState("");

  useEffect(() => {
    const protocol =
      typeof window !== "undefined" && window.location.protocol === "https:"
        ? "wss"
        : "ws";
    const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
    const ws = new WebSocket(`${protocol}://${host}:9090`);

    ws.onmessage = (event) => {
      //console.log("WebSocket message received:", event.data);
      setOutput(event.data);
    };

    ws.onclose = () => {
      //console.log("WebSocket connection closed");
    };

    return () => ws.close();
  }, []);

  return (
    <pre
      style={{
        backgroundColor: "#000",
        color: "#0f0",
        padding: "10px",
        borderRadius: "5px",
      }}
    >
      {output}
    </pre>
  );
};

export default BtopWebSocket;
