# chesshacks

A CLI tool for creating a starter bot for ChessHacks.

## Usage

```bash
npx chesshacks [create/install] [target]
```

### create

```bash
npx chesshacks create <bot-name>
# or
npx chesshacks <bot-name>
```

Creates a new bot in the current directory, with scaffolded starter code and devtools. Keep in mind that devtools are gitignored, so people who clone your repo will need to run `npx chesshacks install` to install devtools.

### install

```bash
npx chesshacks install [target-dir]
```

Installs ChessHacks devtools in the current directory (or in `target-dir` if provided). These devtools will be gitignored.

Once you've initialized your starter app, see its README.md for instructions on how to use it.
