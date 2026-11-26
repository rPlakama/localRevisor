{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.localrevisor;
in
{
  options.services.localrevisor = {
    enable = lib.mkEnableOption "localRevisor ENEM essay grading service";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The localrevisor package to run.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Listen address for the localRevisor WSGI server.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 5000;
      description = "Listen port for the localRevisor WSGI server.";
    };

    dataDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/localrevisor";
      description = "Persistent directory for SQLite DB, uploads, and session secret.";
    };

    bertDir = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional path to the local BERTimbau checkpoint directory.";
    };

    bertMaxTokens = lib.mkOption {
      type = lib.types.int;
      default = 2048;
      description = "Maximum tokens evaluated by BERTimbau sliding windows.";
    };

    maxWorkers = lib.mkOption {
      type = lib.types.int;
      default = 2;
      description = "Maximum concurrent grading worker threads.";
    };

    trustedHosts = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [
        "localhost"
        "127.0.0.1"
        "::1"
        "[::1]"
      ];
      description = "Allowed Host header values for Flask TRUSTED_HOSTS.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional EnvironmentFile path containing OPENAI_API_KEY and SMTP secrets.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.localrevisor = {
      description = "localRevisor ENEM Essay Grading Service";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        LOCALREVISOR_HOST = cfg.host;
        LOCALREVISOR_PORT = toString cfg.port;
        LOCALREVISOR_DATA_DIR = cfg.dataDir;
        LOCALREVISOR_UPLOAD_DIR = "${cfg.dataDir}/uploads";
        LOCALREVISOR_DB_PATH = "${cfg.dataDir}/app.db";
        LOCALREVISOR_ENV_PATH = "${cfg.dataDir}/.env";
        LOCALREVISOR_MAX_WORKERS = toString cfg.maxWorkers;
        BERT_MAX_TOKENS = toString cfg.bertMaxTokens;
        LOCALREVISOR_TRUSTED_HOSTS = lib.concatStringsSep "," cfg.trustedHosts;
      }
      // lib.optionalAttrs (cfg.bertDir != null) {
        BERT_DIR = toString cfg.bertDir;
      };

      serviceConfig = {
        ExecStart = "${lib.getExe cfg.package}";
        DynamicUser = true;
        StateDirectory = "localrevisor";
        StateDirectoryMode = "0700";
        WorkingDirectory = cfg.dataDir;
        Restart = "on-failure";
        RestartSec = "5s";

        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [
          "AF_INET"
          "AF_INET6"
          "AF_UNIX"
        ];
        LockPersonality = true;
      }
      // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };
  };
}
