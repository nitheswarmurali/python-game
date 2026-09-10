import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const output = resolve(root, 'public');
const template = await readFile(resolve(root, 'templates', 'index.html'), 'utf8');
const apiBase = (process.env.VISION_API_URL || '').replace(/\/$/, '');

await mkdir(output, { recursive: true });
await cp(resolve(root, 'static'), resolve(output, 'static'), { recursive: true });
await writeFile(resolve(output, 'index.html'), template, 'utf8');
await writeFile(resolve(output, 'static', 'runtime-config.js'), `window.VISION_API_URL = ${JSON.stringify(apiBase)};\n`, 'utf8');

const indexPath = resolve(output, 'index.html');
const index = await readFile(indexPath, 'utf8');
await writeFile(indexPath, index.replace('</head>', '  <script src="/static/runtime-config.js"></script>\n</head>'), 'utf8');