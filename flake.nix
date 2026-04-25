{
  description = "Development shell for FEASTSolver.jl";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        runtimeLibs = [
          pkgs.stdenv.cc.cc.lib
        ];
        juliaLsp = pkgs.writeShellScriptBin "julia-lsp" ''
          set -euo pipefail

          lsp_project="''${JULIA_LSP_PROJECT:-$PWD/.julia/environments/lsp}"

          if [ ! -f "$lsp_project/Project.toml" ]; then
            echo "julia-lsp: missing LSP environment at $lsp_project" >&2
            echo "Enter the dev shell once to bootstrap LanguageServer.jl." >&2
            exit 1
          fi

          exec julia --project="$lsp_project" --startup-file=no -e 'using LanguageServer; runserver()'
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.julia-bin
            pkgs.git
            pkgs.gnumake
            pkgs.pkg-config
            pkgs.just
            juliaLsp
          ] ++ runtimeLibs;

          JULIA_PROJECT = "@.";
          JULIA_LOAD_PATH = "@:@stdlib";
          JULIA_NUM_THREADS = "auto";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

          shellHook = ''
            export JULIA_DEPOT_PATH="$PWD/.julia:''${JULIA_DEPOT_PATH:-$HOME/.julia}"
            export JULIA_LSP_PROJECT="$PWD/.julia/environments/lsp"

            mkdir -p "$JULIA_LSP_PROJECT"
            if [ ! -f "$JULIA_LSP_PROJECT/Project.toml" ]; then
              echo "Bootstrapping Julia LSP environment in $JULIA_LSP_PROJECT"
              julia --project="$JULIA_LSP_PROJECT" --startup-file=no -e '
                using Pkg
                Pkg.add([
                  PackageSpec(name="LanguageServer"),
                  PackageSpec(name="SymbolServer")
                ])
              '
            fi

            echo "FEASTSolver.jl dev shell"
            echo "  julia project: $JULIA_PROJECT"
            echo "  julia depot:   $JULIA_DEPOT_PATH"
            echo "  julia lsp:     $JULIA_LSP_PROJECT"
          '';
        };
      });
}
