{ pkgs, python, srcRoot }:

let
  appBundle = pkgs.runCommand "localrevisor-bundle" { } ''
    mkdir -p "$out/lib/localrevisor"
    cp -r ${srcRoot}/src "$out/lib/localrevisor/src"
    cp -r ${srcRoot}/ml "$out/lib/localrevisor/ml"
  '';

  localrevisor = pkgs.writeShellApplication {
    name = "localrevisor";
    runtimeInputs = [
      python
      pkgs.ffmpeg
    ];
    text = ''
      export PYTHONPATH="${appBundle}/lib/localrevisor''${PYTHONPATH:+:$PYTHONPATH}"
      for arg in "$@"; do
        case "$arg" in
          benchredação|benchredacao)
            exec python -m src.bench "$@"
            ;;
        esac
      done
      HOST="''${LOCALREVISOR_HOST:-127.0.0.1}"
      PORT="''${LOCALREVISOR_PORT:-5000}"
      if [ "''${LOCALREVISOR_DEV:-0}" = "1" ]; then
        exec python -m flask --app src.main run --debug --host "$HOST" --port "$PORT" "$@"
      fi
      exec python -m waitress --host="$HOST" --port="$PORT" src.main:app
    '';
  };

  mlPipeline = pkgs.writeShellApplication {
    name = "ml-pipeline";
    runtimeInputs = [ python ];
    text = ''
      export PYTHONPATH="${appBundle}/lib/localrevisor''${PYTHONPATH:+:$PYTHONPATH}"
      exec python -m ml.pipeline "$@"
    '';
  };
in
{
  inherit localrevisor mlPipeline;
}
