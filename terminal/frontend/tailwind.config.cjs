/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'terminal-black': '#0a0a0a',
                'terminal-green': '#00ff41',
                'terminal-red': '#ff0033',
                'terminal-gray': '#1a1a1a',
            },
            fontFamily: {
                mono: ['Menlo', 'Monaco', 'Courier New', 'monospace'],
            }
        },
    },
    plugins: [],
}
