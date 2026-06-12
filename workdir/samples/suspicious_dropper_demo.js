// Benign static-inspection demo sample.
// The string values below are documentation-only indicators.

const demoUrl = "http://example.invalid/c2/check";
const encodedUrl = "aHR0cDovL2V4YW1wbGUuaW52YWxpZC9jMi9jaGVjaw==";
const testIp = "192.0.2.45";
const commandHint = "powershell -nop -w hidden -enc <demo>";

function decodeDemoValue(value) {
  return "decoded:" + value;
}

function suspiciousApiReferences() {
  const apiNames = [
    "atob",
    "String.fromCharCode",
    "XMLHttpRequest",
    "eval",
    "WScript.Shell",
  ];
  return apiNames.join(",");
}

console.log("demo only", demoUrl, encodedUrl, testIp, commandHint, decodeDemoValue("safe"), suspiciousApiReferences());
