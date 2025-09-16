# 改行コードの設定を決めました

## 1. Mac/windows共通設定

git config --global core.autocrlf false   # 変換は .gitattributes に任せる

git config --global core.safecrlf warn    # 変な混在改行を警告

git config --global core.filemode false   # 実行権限の差分を無視（誤検知防止）

git config --global pull.rebase false     # まずは安全に merge pull（運用で決めてOK）

git config --global fetch.prune true      # 消えたリモートブランチを掃除

git config --global init.defaultBranch main

## 2. windows設定

git config --global core.ignorecase true      # 既定で true。大文字小文字違いの衝突を回避

git config --global credential.helper manager # Git Credential Manager

## 3. macOS設定

git config --global credential.helper osxkeychain

git config --global core.precomposeunicode true  # mac のNFD問題に強い

## 4. エディタ設定(できれば)

root = true

end_of_line = lf

insert_final_newline = true

charset = utf-8

indent_style = space

indent_size = 2

[*.ps1]
end_of_line = crlf

[*.bat]
end_of_line = crlf

