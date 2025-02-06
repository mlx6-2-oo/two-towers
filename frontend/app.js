// Configure this to point to your backend server
const SERVER_URL = 'http://65.109.80.113:8000';

// Initialize terminal
const term = new Terminal({
    cursorBlink: true,
    theme: {
        background: '#1e1e1e',
        foreground: '#ffffff'
    },
    fontSize: 16
});

// Create and load FitAddon
const fitAddon = new FitAddon.FitAddon();
term.loadAddon(fitAddon);

// Open terminal and fit to screen
term.open(document.getElementById('terminal'));
fitAddon.fit();

// Handle window resizing
window.addEventListener('resize', () => {
    fitAddon.fit();
});

term.write('\n$ ');

let inputBuffer = '';

term.onData(e => {
    switch (e) {
        case '\r': // Enter
            term.write('\n');
            handleCommand(inputBuffer);
            inputBuffer = '';
            break;
        case '\u007F': // Backspace
            if (inputBuffer.length > 0) {
                inputBuffer = inputBuffer.slice(0, -1);
                term.write('\b \b');
            }
            break;
        default:
            inputBuffer += e;
            term.write(e);
    }
});

async function handleCommand(command) {
    if (command.toLowerCase() === 'quit') {
        term.writeln('Goodbye!');
        return;
    }

    try {
        term.writeln('Searching...');
        const response = await fetch(`${SERVER_URL}/search/?query=${encodeURIComponent(command)}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const text = await response.text();
        console.log('Raw response:', text);
        
        let results;
        try {
            results = JSON.parse(text);
        } catch (e) {
            throw new Error(`Failed to parse JSON: ${text}`);
        }
        
        term.writeln('\nResults:');
        results.forEach((r, i) => {
            term.writeln(`\n${i + 1}. Score: ${r.similarity.toFixed(3)}`);
            term.writeln(`ID: ${r.id}`);
            term.writeln(`${r.document.slice(0, 200)}...`);
        });
    } catch (error) {
        term.writeln(`Error: ${error.message}`);
        console.error('Full error:', error);
    }

    term.write('\n$ ');
} 