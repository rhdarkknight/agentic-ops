/*
    Hack tools / offensive security indicators.
    Severity: HIGH. Note: not always malicious (pentesters use them)
    but should never be present in production skills.
*/

rule hacktool_sqlmap
{
    meta:
        description = "sqlmap SQL injection tool"
        category = "hacktool"
        severity = "HIGH"
        confidence = "0.95"
    strings:
        $s1 = "sqlmap/" nocase
        $s2 = "sqlmap.py" nocase
        $s3 = "tamper_scripts" nocase
    condition:
        any of them
}

rule hacktool_nmap
{
    meta:
        description = "nmap / masscan network scanner (allowed in authorized pentests only)"
        category = "hacktool"
        severity = "MEDIUM"
        confidence = "0.7"
    strings:
        $n1 = "nmap -sS" nocase
        $n2 = "nmap --script" nocase
        $n3 = "masscan" nocase
    condition:
        any of them
}

rule hacktool_mimikatz
{
    meta:
        description = "Mimikatz credential dumper"
        category = "hacktool"
        severity = "CRITICAL"
        confidence = "0.95"
    strings:
        $m1 = "mimikatz" nocase
        $m2 = "sekurlsa::logonpasswords" nocase
        $m3 = "lsadump::sam" nocase
    condition:
        any of them
}

rule hacktool_bloodhound
{
    meta:
        description = "BloodHound AD attack-path mapper"
        category = "hacktool"
        severity = "HIGH"
        confidence = "0.85"
    strings:
        $bh1 = "bloodhound" nocase
        $bh2 = "SharpHound" nocase
    condition:
        any of them
}

rule hacktool_burp
{
    meta:
        description = "Burp Suite proxy / scanner"
        category = "hacktool"
        severity = "MEDIUM"
        confidence = "0.7"
    strings:
        $b1 = "burpsuite" nocase
        $b2 = "burp-collaborator" nocase
    condition:
        any of them
}
