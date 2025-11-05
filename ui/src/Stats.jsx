import React, { useState } from "react";
import Container from "react-bootstrap/Container";
import Tabs from "react-bootstrap/Tabs";
import Tab from "react-bootstrap/Tab";
//import BtopWebSocket from "./BtopWebSocket";
//import TerminalTab from "./TerminalTab";
import BtopTab from "./BtopTab";

const Stats = () => {
  const [activeKey, setActiveKey] = useState("btop");

  return (
    <Container className="mt-4">
      <h1>Stats Dashboard</h1>
      <Tabs
        id="stats-tabs"
        activeKey={activeKey}
        onSelect={(key) => setActiveKey(key)}
        className="mb-3"
      >
        <Tab eventKey="btopterminal" title="btop">
          <BtopTab />
        </Tab>
      </Tabs>
    </Container>
  );
};

export default Stats;
