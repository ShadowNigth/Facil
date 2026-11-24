# Modelos para probar

Los `.mdl` ya tienen las **texturas embebidas** (8-bit BMP dentro del propio
archivo), así que no hacen falta `.bmp` sueltos ni `.t.mdl`.

| archivo | modelo | tamaño | huesos | tris | secuencias |
|---|---|---|---|---|---|
| `v_glock18.mdl` | Glock 18 | 1.87 MB | 67 | 4 099 | 21 |
| `v_ak47.mdl` | AK-47 | 1.88 MB | 70 | 4 345 | 20 |

## Instalación

Copiá los `.mdl` a tu carpeta de modelos del juego, respetando la ruta:

```
<tu_cstrike>/models/v_glock18.mdl
<tu_cstrike>/models/v_ak47.mdl
```

En un servidor dedicado la ruta es la misma dentro de `cstrike/`. Si jugás en
un servidor con *sv_consistency 1* y los modelos no son los oficiales, el
cliente te los va a rechazar: probá en un servidor local o con `sv_consistency 0`.

## Cómo verlos

Los modelos son **viewmodels** (`v_`), o sea armas en primera persona:

1. En el juego, `cstrike/models/v_glock18.mdl` se usa automáticamente si reemplaza
   al original, o
2. probalos con un visor: HL Model Viewer, Jed's Half-Life Model Viewer, o
   abrilos en un mapa con un `weapon_glock18` custom.

Para verlos sin el juego, el visor que escribí en este repo los rasteriza:

```
python tools/mdlpreview.py out/v_glock18.mdl preview.png
```

## Qué mirar (y qué puede fallar)

Lo que ya está verificado en el `.mdl`:

* bind pose fiel al origen (desvío mediano 0.35 unidades, máx 1.37)
* ninguna secuencia estira la malla (arista máxima = la del bind)
* el arma se mantiene en las manos y con tamaño estable en los 21 clips

Lo que **no** puedo verificar sin el motor real:

* que el juego los cargue y los muestre bien
* el ángulo/posición final en pantalla, que depende del `attachment` y de la
  config del viewmodel, no del modelo en sí
* si el espejo global (el modelo es la imagen espejada del original: la mano
  derecha del modelo agarra con la geometría que en el origen era la izquierda)
  se nota visualmente

Decime qué ves y con eso ajusto.
