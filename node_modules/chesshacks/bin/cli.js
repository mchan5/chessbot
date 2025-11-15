#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const rawArgs = process.argv.slice(2);

const defaultGitignore =
  "__pycache__\n.venv\nvenv\n.env.local\n.env\ndevtools\n";

let command = "create";
let projectName;

if (rawArgs[0] === "create" || rawArgs[0] === "install") {
  command = rawArgs[0];
  projectName = rawArgs[1];
} else {
  projectName = rawArgs[0];
}

if (!projectName) {
  projectName = "my-chesshacks-bot";
}

const projectPath = path.join(process.cwd(), projectName);
const starterPath = path.join(__dirname, "starter");

fs.mkdirSync(projectPath, { recursive: true });

function copyRecursive(src, dest) {
  const stats = fs.statSync(src);

  if (stats.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src);
    for (const entry of entries) {
      const srcEntry = path.join(src, entry);
      const destEntry = path.join(dest, entry);
      copyRecursive(srcEntry, destEntry);
    }
  } else {
    const destDir = path.dirname(dest);
    fs.mkdirSync(destDir, { recursive: true });
    const contents = fs.readFileSync(src);
    fs.writeFileSync(dest, contents);
  }
}

if (command === "install") {
  const installRoot = projectName ? projectPath : process.cwd();
  const devtoolsSrc = path.join(starterPath, "devtools");
  const devtoolsDest = path.join(installRoot, "devtools");

  fs.mkdirSync(installRoot, { recursive: true });
  copyRecursive(devtoolsSrc, devtoolsDest);

  const gitignorePath = path.join(installRoot, ".gitignore");
  let gitignoreContents = "";
  if (fs.existsSync(gitignorePath)) {
    gitignoreContents = fs.readFileSync(gitignorePath, "utf8");
  }

  if (
    !gitignoreContents.split("\n").some((line) => line.trim() === "devtools")
  ) {
    fs.writeFileSync(gitignorePath, defaultGitignore);
  }

  console.log(`\x1b[32m\x1b[1m\x1b[0m Installed devtools in ${installRoot}`);
} else {
  copyRecursive(starterPath, projectPath);
  const gitignorePath = path.join(projectPath, ".gitignore");
  let gitignoreContents = "";
  if (fs.existsSync(gitignorePath)) {
    gitignoreContents = fs.readFileSync(gitignorePath, "utf8");
  }

  if (
    !gitignoreContents.split("\n").some((line) => line.trim() === "devtools")
  ) {
    fs.writeFileSync(gitignorePath, defaultGitignore);
  }

  console.log(`\x1b[32m\x1b[1m\x1b[0m Created ${projectName}`);
}
