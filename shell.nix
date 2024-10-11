{ inputs, self, ... }:
{
  perSystem =
    { pkgs, self', ... }:
    {

      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          pyright 
          nixd
        ];
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

          if [ ! -d .venv ]; then
            python3 -m venv .venv
          fi
          source .venv/bin/activate
        '';
      };

    };
}
