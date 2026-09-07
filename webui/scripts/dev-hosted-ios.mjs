import { randomBytes } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import {
  access,
  chmod,
  mkdir,
  mkdtemp,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { networkInterfaces, hostname } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webuiRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(webuiRoot, "..");
const httpsDirectory = resolve(
  repositoryRoot,
  "browser-runtime",
  ".local-build",
  "https",
);
const caKeyPath = join(httpsDirectory, "drowse-local-ca.key.pem");
const caCertificatePath = join(httpsDirectory, "drowse-local-ca.cert.pem");
const caInstallPath = join(httpsDirectory, "drowse-local-ca.cer");
const serverKeyPath = join(httpsDirectory, "drowse-ios.key.pem");
const serverCertificatePath = join(httpsDirectory, "drowse-ios.cert.pem");
const vitePath = resolve(webuiRoot, "node_modules", "vite", "bin", "vite.js");

function usage() {
  return `Usage: npm run dev:hosted:ios -- [--port <port>] [--ip <LAN IPv4>]

Creates a private local development CA, issues a certificate for this Mac's LAN
addresses, and runs the hosted Drowse app over HTTPS on 0.0.0.0.

Options:
  --port <port>  Vite port (default: 4173)
  --ip <address> Use this LAN IPv4 instead of automatic discovery
  --help          Show this help`;
}

function fail(message) {
  throw new Error(message);
}

function parseArguments(arguments_) {
  let port = 4173;
  let ip;

  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--help" || argument === "-h") return { help: true };
    if (argument === "--port" || argument === "--ip") {
      const value = arguments_[index + 1];
      if (!value || value.startsWith("--")) fail(`${argument} requires a value`);
      if (argument === "--port") port = Number(value);
      else ip = value;
      index += 1;
      continue;
    }
    if (argument.startsWith("--port=")) {
      port = Number(argument.slice("--port=".length));
      continue;
    }
    if (argument.startsWith("--ip=")) {
      ip = argument.slice("--ip=".length);
      continue;
    }
    fail(`Unknown option: ${argument}\n\n${usage()}`);
  }

  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    fail("--port must be an integer from 1024 through 65535");
  }
  if (ip && !isIPv4(ip)) fail("--ip must be a valid IPv4 address");
  return { help: false, port, ip };
}

function isIPv4(value) {
  const parts = value.split(".");
  return (
    parts.length === 4 &&
    parts.every(
      (part) => /^\d{1,3}$/.test(part) && Number(part) >= 0 && Number(part) <= 255,
    )
  );
}

function isLanIPv4(value) {
  if (!isIPv4(value)) return false;
  const [a, b] = value.split(".").map(Number);
  return (
    a === 10 ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 100 && b >= 64 && b <= 127)
  );
}

function defaultRouteInterface() {
  const result = spawnSync("route", ["-n", "get", "default"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"],
  });
  if (result.status !== 0) return undefined;
  return /^\s*interface:\s*(\S+)/m.exec(result.stdout)?.[1];
}

function discoverLanAddresses(explicitIp) {
  if (explicitIp) return [{ name: "manual", address: explicitIp }];

  const defaultInterface = defaultRouteInterface();
  const virtualInterface = /^(?:lo|utun|awdl|llw|bridge|docker|veth|vmnet|vbox|tailscale|wg)/i;
  const candidates = [];
  for (const [name, addresses] of Object.entries(networkInterfaces())) {
    if (!addresses || virtualInterface.test(name)) continue;
    for (const address of addresses) {
      if (
        address.family === "IPv4" &&
        !address.internal &&
        isLanIPv4(address.address)
      ) {
        candidates.push({ name, address: address.address });
      }
    }
  }

  candidates.sort((left, right) => {
    const score = ({ name }) =>
      (name === defaultInterface ? 0 : 100) +
      (/^en\d+$/i.test(name) ? 0 : /^eth\d*$/i.test(name) ? 10 : 20);
    return score(left) - score(right) || left.name.localeCompare(right.name);
  });

  const unique = candidates.filter(
    (candidate, index) =>
      candidates.findIndex(({ address }) => address === candidate.address) === index,
  );
  if (unique.length === 0) {
    fail(
      "No private LAN IPv4 address was found. Connect this Mac to the same Wi-Fi as the iPhone, disable an interfering VPN, or pass --ip <address>.",
    );
  }
  return unique;
}

function localHostnames() {
  const safeHostname = hostname()
    .toLowerCase()
    .replace(/[^a-z0-9.-]+/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "");
  if (!safeHostname || !/[a-z0-9]/.test(safeHostname)) return ["localhost"];
  const names = ["localhost", safeHostname];
  if (!safeHostname.endsWith(".local")) names.push(`${safeHostname}.local`);
  return [...new Set(names)];
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function runOpenSsl(label, arguments_) {
  const result = spawnSync("openssl", arguments_, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.error?.code === "ENOENT") {
    fail("OpenSSL is required. Install it with Xcode Command Line Tools or Homebrew, then retry.");
  }
  if (result.error) fail(`${label} failed: ${result.error.message}`);
  if (result.status !== 0) {
    const detail = result.stderr.trim().split("\n").at(-1);
    fail(`${label} failed${detail ? `: ${detail}` : ""}`);
  }
  return result.stdout;
}

async function validateExistingCa() {
  const [hasKey, hasCertificate] = await Promise.all([
    exists(caKeyPath),
    exists(caCertificatePath),
  ]);
  if (hasKey !== hasCertificate) {
    fail(
      `The local CA is incomplete. Remove ${httpsDirectory} and run this command again, then reinstall the new CA on the iPhone.`,
    );
  }
  if (!hasKey) return false;

  const keyPublic = runOpenSsl("Reading the local CA key", [
    "pkey",
    "-in",
    caKeyPath,
    "-pubout",
  ]).trim();
  const certificatePublic = runOpenSsl("Reading the local CA certificate", [
    "x509",
    "-in",
    caCertificatePath,
    "-pubkey",
    "-noout",
  ]).trim();
  if (keyPublic !== certificatePublic) {
    fail(
      `The local CA key and certificate do not match. Remove ${httpsDirectory} and retry, then reinstall the new CA on the iPhone.`,
    );
  }
  runOpenSsl("Checking the local CA lifetime", [
    "x509",
    "-in",
    caCertificatePath,
    "-checkend",
    "86400",
    "-noout",
  ]);
  return true;
}

async function createCertificates(addresses) {
  await mkdir(httpsDirectory, { recursive: true, mode: 0o700 });
  await chmod(httpsDirectory, 0o700);
  const temporaryDirectory = await mkdtemp(join(httpsDirectory, ".certificate-"));

  try {
    if (!(await validateExistingCa())) {
      const temporaryCaKey = join(temporaryDirectory, "ca.key.pem");
      const temporaryCaCertificate = join(temporaryDirectory, "ca.cert.pem");
      const caConfig = join(temporaryDirectory, "ca.cnf");
      await writeFile(
        caConfig,
        `[req]
distinguished_name = dn
x509_extensions = v3_ca
prompt = no

[dn]
CN = Drowse local development CA

[v3_ca]
basicConstraints = critical, CA:TRUE, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
`,
        { mode: 0o600 },
      );
      runOpenSsl("Creating the local development CA", [
        "req",
        "-x509",
        "-newkey",
        "rsa:3072",
        "-nodes",
        "-sha256",
        "-days",
        "3650",
        "-config",
        caConfig,
        "-keyout",
        temporaryCaKey,
        "-out",
        temporaryCaCertificate,
      ]);
      await chmod(temporaryCaKey, 0o600);
      await rename(temporaryCaKey, caKeyPath);
      await rename(temporaryCaCertificate, caCertificatePath);
    }

    const names = localHostnames();
    const altNames = [
      ...names.map((name, index) => `DNS.${index + 1} = ${name}`),
      "IP.1 = 127.0.0.1",
      ...addresses.map(({ address }, index) => `IP.${index + 2} = ${address}`),
    ].join("\n");
    const serverConfig = join(temporaryDirectory, "server.cnf");
    const temporaryServerKey = join(temporaryDirectory, "server.key.pem");
    const temporaryServerCertificate = join(temporaryDirectory, "server.cert.pem");
    const serverRequest = join(temporaryDirectory, "server.csr.pem");
    await writeFile(
      serverConfig,
      `[req]
distinguished_name = dn
prompt = no

[dn]
CN = ${addresses[0].address}

[v3_server]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
${altNames}
`,
      { mode: 0o600 },
    );
    runOpenSsl("Creating the HTTPS private key", [
      "req",
      "-new",
      "-newkey",
      "rsa:2048",
      "-nodes",
      "-sha256",
      "-config",
      serverConfig,
      "-keyout",
      temporaryServerKey,
      "-out",
      serverRequest,
    ]);
    runOpenSsl("Issuing the HTTPS certificate", [
      "x509",
      "-req",
      "-in",
      serverRequest,
      "-CA",
      caCertificatePath,
      "-CAkey",
      caKeyPath,
      "-set_serial",
      `0x${randomBytes(16).toString("hex")}`,
      "-days",
      "365",
      "-sha256",
      "-extfile",
      serverConfig,
      "-extensions",
      "v3_server",
      "-out",
      temporaryServerCertificate,
    ]);
    runOpenSsl("Verifying the HTTPS certificate", [
      "verify",
      "-CAfile",
      caCertificatePath,
      temporaryServerCertificate,
    ]);
    await chmod(temporaryServerKey, 0o600);
    await rename(temporaryServerKey, serverKeyPath);
    await rename(temporaryServerCertificate, serverCertificatePath);

    const temporaryInstallCertificate = join(temporaryDirectory, "ca.cer");
    runOpenSsl("Preparing the iPhone CA certificate", [
      "x509",
      "-in",
      caCertificatePath,
      "-outform",
      "der",
      "-out",
      temporaryInstallCertificate,
    ]);
    await rename(temporaryInstallCertificate, caInstallPath);
    await Promise.all([
      chmod(caKeyPath, 0o600),
      chmod(serverKeyPath, 0o600),
      chmod(caCertificatePath, 0o644),
      chmod(serverCertificatePath, 0o644),
      chmod(caInstallPath, 0o644),
    ]);
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

function printInstructions(addresses, port) {
  const urls = addresses.map(({ address }) => `https://${address}:${port}/app`);
  console.log(`
Drowse will be available to devices on this network:
${urls.map((url) => `  ${url}`).join("\n")}

Set up the iPhone once:
  1. AirDrop this public certificate to the iPhone and accept it:
     ${caInstallPath}
  2. Open Settings > General > VPN & Device Management, select the downloaded
     Drowse local development CA profile, and tap Install.
  3. Open Settings > General > About > Certificate Trust Settings and enable
     full trust for Drowse local development CA.
  4. Open one of the URLs above in Safari. The Mac and iPhone must be on the
     same Wi-Fi network.
  5. To install Drowse, use Safari's Share > Add to Home Screen > Add.

Private keys remain in the ignored local build directory and are not printed.
Press Ctrl+C to stop the server.
`);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  if (!(await exists(vitePath))) {
    fail(`Vite is not installed in ${webuiRoot}. Run npm ci in that directory first.`);
  }

  runOpenSsl("Checking OpenSSL", ["version"]);
  const addresses = discoverLanAddresses(options.ip);
  await createCertificates(addresses);
  printInstructions(addresses, options.port);

  const server = spawn(
    process.execPath,
    [
      vitePath,
      "--config",
      resolve(webuiRoot, "vite.hosted.config.ts"),
      "--host",
      "0.0.0.0",
      "--port",
      String(options.port),
      "--strictPort",
    ],
    {
      cwd: webuiRoot,
      env: {
        ...process.env,
        DROWSE_HTTPS_CERT: serverCertificatePath,
        DROWSE_HTTPS_KEY: serverKeyPath,
      },
      stdio: "inherit",
    },
  );

  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => server.kill(signal));
  }
  const result = await new Promise((resolveExit, rejectExit) => {
    server.once("error", rejectExit);
    server.once("exit", (code, signal) => resolveExit({ code, signal }));
  });
  process.exitCode = result.code ?? (result.signal === "SIGINT" ? 130 : 1);
}

main().catch((error) => {
  console.error(`Drowse iPhone HTTPS setup failed: ${error.message}`);
  process.exitCode = 1;
});
