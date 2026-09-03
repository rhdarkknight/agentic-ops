/*
    Crypto miner detection rules.
    Patterns: Neo23x0/signature-base + current threat intel.
    Severity: HIGH (resource hijack), Confidence: 0.9
*/

rule crypto_stratum_protocol
{
    meta:
        description = "Stratum mining protocol usage (stratum+tcp/ssl, mining.subscribe/authorize)"
        category = "cryptominer"
        severity = "HIGH"
        confidence = "0.9"
    strings:
        $stratum_tcp  = "stratum+tcp://" nocase
        $stratum_ssl  = "stratum+ssl://" nocase
        $mining_sub   = "mining.subscribe" nocase
        $mining_auth  = "mining.authorize" nocase
        $mining_submit = "mining.submit" nocase
    condition:
        any of them
}

rule crypto_mining_pools
{
    meta:
        description = "Connection to known cryptocurrency mining pools"
        category = "cryptominer"
        severity = "HIGH"
        confidence = "0.85"
    strings:
        $pool1 = "pool.minexmr.com" nocase
        $pool2 = "xmrpool.eu" nocase
        $pool3 = "nanopool.org" nocase
        $pool4 = "moneroocean.stream" nocase
        $pool5 = "hashvault.pro" nocase
        $pool6 = "supportxmr.com" nocase
    condition:
        any of them
}

rule crypto_miner_binary
{
    meta:
        description = "Known cryptominer binary name or class"
        category = "cryptominer"
        severity = "HIGH"
        confidence = "0.9"
    strings:
        $b1 = "xmrig" nocase
        $b2 = "minerd" nocase
        $b3 = "cpuminer" nocase
        $b4 = "ethminer" nocase
        $b5 = "claymore" nocase
        $b6 = "phoenixminer" nocase
        $b7 = "cryptonight" nocase
    condition:
        any of them
}
