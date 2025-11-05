const { spawn } = require('node-pty');
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 9091 });

wss.on('connection', (ws) => {
    const shell = spawn(process.env.SHELL || 'bash', [], {
        name: 'xterm-color',
        cols: 80,
        rows: 80,
        cwd: process.env.HOME,
        env: process.env,
    });

    // Send shell output to the client
    shell.on('data', (data) => {
        ws.send(data);
    });

    // Send client input to the shell
    ws.on('message', (message) => {
        shell.write(message);
    });

    // Clean up on close
    ws.on('close', () => {
        shell.kill();
    });
});

console.log('WebSocket server running on ws://localhost:9091');
