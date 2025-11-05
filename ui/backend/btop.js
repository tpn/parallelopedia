const { spawn } = require('node-pty');
const express = require('express');
const WebSocket = require('ws');

const app = express();
const port = 9090;

// WebSocket clients
const clients = new Set();

// Start `top` or `btop` command
const command = 'zsh'; // Change to 'btop' if needed
const ptyProcess = spawn(command, ['-c', 'btop'], {
    name: 'xterm',
    cols: 344, // Match the frontend terminal dimensions
    rows: 84,
    cwd: process.env.HOME,
    env: process.env,
});

var firstOutput = '';

// Capture and broadcast terminal output
ptyProcess.on('data', (data) => {
    const output = data.toString();
    if (firstOutput == '') {
        firstOutput = output;
    }
    //console.log(output);
    // Broadcast to all connected WebSocket clients
    clients.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(output);
        }
    });
});

// Cleanup on server exit
process.on('SIGINT', () => {
    ptyProcess.kill();
    process.exit();
});

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ noServer: true });

wss.on('connection', (ws) => {
    // Send initial output.
    if (firstOutput != '') {
        ws.send(firstOutput);
    }

    clients.add(ws);

    ws.on('close', () => {
        clients.delete(ws);
    });
});

// HTTP server for WebSocket upgrades
const server = app.listen(port, () => {
    console.log(`Server listening on http://localhost:${port}`);
});

server.on('upgrade', (req, socket, head) => {
    wss.handleUpgrade(req, socket, head, (ws) => {
        wss.emit('connection', ws, req);
    });
});
