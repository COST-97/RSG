from sklearn.manifold import TSNE as skTSNE
import trimap
import umap.umap_ as umap
import pacmap


class UMAP:

    def __init__(self) -> None:
        self._umap = umap.UMAP(random_state=1,)

    def forward(self, embeds):
        return self._umap.fit_transform(embeds)


class TSNE:

    def __init__(self) -> None:
        self._tsne = skTSNE(n_components=2, random_state=1)

    def forward(self, embeds):
        return self._tsne.fit_transform(embeds)


class TriMAP:

    def __init__(self) -> None:
        self._trimap = trimap.TRIMAP()

    def forward(self, embeds):
        return self._trimap.fit_transform(embeds)


class PacMAP:

    def __init__(self) -> None:
        self._pacmap = pacmap.PaCMAP(random_state=1)

    def forward(self, embeds):
        return self._pacmap.fit_transform(embeds)