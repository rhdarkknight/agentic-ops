/*
    Webshell detection rules.
    Patterns: PHP/Python/JSP/ASPX webshell signatures.
    Severity: HIGH
*/

rule php_webshell_eval
{
    meta:
        description = "PHP webshell using eval() with request input"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.9"
    strings:
        $eval1 = "eval($_GET" nocase
        $eval2 = "eval($_POST" nocase
        $eval3 = "eval($_REQUEST" nocase
        $eval4 = "@eval($" nocase
    condition:
        any of them
}

rule php_webshell_system
{
    meta:
        description = "PHP webshell invoking system/exec/passthru"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.85"
    strings:
        $sys1 = "system($_GET" nocase
        $sys2 = "system($_POST" nocase
        $sys3 = "passthru($_GET" nocase
        $sys4 = "shell_exec($_GET" nocase
        $sys5 = "exec($_GET" nocase
        $sys6 = "popen($_GET" nocase
    condition:
        any of them
}

rule php_webshell_encoded
{
    meta:
        description = "PHP base64-encoded payload execution"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.95"
    strings:
        $b64 = "base64_decode(" nocase
        $b64eval = "eval(base64_decode" nocase
        $b64create = "create_function(" nocase
    condition:
        $b64 and ($b64eval or $b64create)
}

rule python_webshell
{
    meta:
        description = "Python reverse shell or one-liner webshell"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.85"
    strings:
        $py1 = "import os; os.system" nocase
        $py2 = "subprocess.Popen(['/bin/sh" nocase
        $py3 = "pty.spawn('/bin/sh')" nocase
        $py4 = "__import__('os').system" nocase
        $py5 = "exec(request" nocase
    condition:
        any of them
}

rule jsp_webshell
{
    meta:
        description = "JSP webshell using Runtime.exec"
        category = "webshell"
        severity = "HIGH"
        confidence = "0.9"
    strings:
        $jsp1 = "Runtime.getRuntime().exec(request" nocase
        $jsp2 = "Process p = Runtime.getRuntime" nocase
        $jsp3 = "Class.forName(\"javax.script" nocase
    condition:
        any of them
}
