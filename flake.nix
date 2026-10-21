{
  description = "Nix-develop-flake for LocalRevisor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python314
          ruff
          uv
          vscode-html-languageserver
          vscode-css-languageserver
        ];
      };
    };
}
