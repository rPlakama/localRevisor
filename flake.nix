{
  description = "Nix-develop-flake and NixOS service module for LocalRevisor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python314.withPackages (
        ps: with ps; [
          # app
          flask
          httpx
          openai
          pypdf
          python-dotenv
          waitress
          # ML pipeline & BERT (opcional)
          joblib
          numpy
          pandas
          scikit-learn
          tqdm
          torch
          transformers
          safetensors
          tokenizers
          huggingface-hub
        ]
      );
      built = import ./nix/package.nix {
        inherit pkgs python;
        srcRoot = self;
      };
      docsServe = pkgs.writeShellApplication {
        name = "docs-serve";
        runtimeInputs = [
          pkgs.mdbook
          pkgs.mdbook-mermaid
        ];
        text = ''
          if [ ! -f DOCS.md ]; then
            echo "Erro: execute a partir da raiz do repositório localRevisor (onde fica DOCS.md)." >&2
            exit 1
          fi

          workdir="$(mktemp -d)"
          trap 'rm -rf "$workdir"' EXIT

          mkdir -p "$workdir/src"
          ln -s "$PWD/DOCS.md" "$workdir/src/DOCS.md"
          printf '# Sumário\n\n- [Pipeline e Arquitetura (DOCS)](DOCS.md)\n' > "$workdir/src/SUMMARY.md"
          if [ -f "$PWD/README.md" ]; then
            ln -s "$PWD/README.md" "$workdir/src/README.md"
            printf -- '- [README](README.md)\n' >> "$workdir/src/SUMMARY.md"
          fi

          cat > "$workdir/book.toml" <<'EOF'
          [book]
          title = "localRevisor — Documentação"
          language = "pt-BR"
          src = "src"

          [output.html]
          default-theme = "light"
          preferred-dark-theme = "navy"
          EOF

          mdbook-mermaid install "$workdir" >/dev/null
          mdbook serve "$workdir" -n 127.0.0.1 "$@"
        '';
      };
    in
    {
      packages.${system} = {
        default = built.localrevisor;
        ml-pipeline = built.mlPipeline;
        DOCS = docsServe;
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${built.localrevisor}/bin/localrevisor";
        };
        ml-pipeline = {
          type = "app";
          program = "${built.mlPipeline}/bin/ml-pipeline";
        };
        DOCS = {
          type = "app";
          program = "${docsServe}/bin/docs-serve";
        };
      };

      nixosModules = {
        default = import ./nix/module.nix;
        localrevisor = import ./nix/module.nix;
      };

      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [
            python
            built.localrevisor
            built.mlPipeline
          ]
          ++ (with pkgs; [
            python314Packages.pytest
            ruff
            ffmpeg
            uv
            vscode-html-languageserver
            vscode-css-languageserver
          ]);

          env = {
            FLASK_APP = "src.main";
          };

          shellHook = ''
            export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };

        DOCS = pkgs.mkShell {
          packages = [
            docsServe
            pkgs.mdbook
            pkgs.mdbook-mermaid
            pkgs.glow
          ];

          shellHook = ''
            echo "Ambiente de documentação (DOCS.md + Mermaid) pronto:"
            echo "  docs-serve        # abre preview HTML com Mermaid em http://127.0.0.1:3000"
            echo "  docs-serve --open # abre automaticamente no navegador"
            echo "  glow DOCS.md      # leitura rápida no terminal"
          '';
        };
      };
    };
}
