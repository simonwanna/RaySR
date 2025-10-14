from sionna.rt import load_scene, Camera
import sionna.rt as rt

class MyScene:
    """
    Wrapper around sionna.rt.Scene that:
      - exposes xmin/xmax/ymin/... on the wrapper,
      - forwards all other attribute gets/sets to the underlying Scene.
    """

    # Optional: keep a clear list of wrapper-owned fields so we don't forward them.
    _OWN_FIELDS = {
        "_scene", "margin", "bbox",
        "raw_xmin","raw_xmax","raw_ymin","raw_ymax","raw_zmin","raw_zmax",
        "xmin","xmax","ymin","ymax","zmin","zmax",
        "top_down_camera","side_camera","diag_camera"
    }

    def __init__(self, filename: str, merge_shapes=False, margin: float = 15.0):
        # IMPORTANT: during __init__, set wrapper fields using object.__setattr__
        object.__setattr__(self, "_scene", load_scene(filename=filename, merge_shapes=merge_shapes))
        object.__setattr__(self, "margin", float(margin))

        bbox = self._scene.mi_scene.bbox()
        object.__setattr__(self, "bbox", bbox)

        # raw bounds
        object.__setattr__(self, "raw_xmin", float(bbox.min.x))
        object.__setattr__(self, "raw_xmax", float(bbox.max.x))
        object.__setattr__(self, "raw_ymin", float(bbox.min.y))
        object.__setattr__(self, "raw_ymax", float(bbox.max.y))
        object.__setattr__(self, "raw_zmin", float(0.0))  # or float(bbox.min.z) if you prefer
        object.__setattr__(self, "raw_zmax", float(bbox.max.z))

        # margined bounds
        xmin = self.raw_xmin + self.margin
        xmax = self.raw_xmax - self.margin
        ymin = self.raw_ymin + self.margin
        ymax = self.raw_ymax - self.margin
        zmin = self.raw_zmin + self.margin
        zmax = self.raw_zmax

        if xmin > xmax or ymin > ymax:
            raise ValueError("Margin too large for scene extents.")

        object.__setattr__(self, "xmin", xmin)
        object.__setattr__(self, "xmax", xmax)
        object.__setattr__(self, "ymin", ymin)
        object.__setattr__(self, "ymax", ymax)
        object.__setattr__(self, "zmin", zmin)
        object.__setattr__(self, "zmax", zmax)

        # camera presets
        try:
            if filename == rt.scene.simple_street_canyon:
                z_top = 300
                y_side, z_side = -250, 150
                x_diag, y_diag, z_diag = self.raw_xmin * 2.3, self.raw_ymax * 2.3, 150
            elif filename == rt.scene.etoile:
                z_top = 1500
                y_side, z_side = -1000, 500
                x_diag, y_diag, z_diag = self.raw_xmin * 2.3, self.raw_ymax * 2.3, 600
            else:
                raise NotImplementedError(
                    f"Camera presets not yet implemented for '{filename}'. "
                    "Presets set to None."
                )
        except NotImplementedError as e:
            print(e)
            object.__setattr__(self, "top_down_camera", None)
            object.__setattr__(self, "side_camera", None)
            object.__setattr__(self, "diag_camera", None)
        else:
            object.__setattr__(self, "top_down_camera", Camera(position=[0, 0, z_top], look_at=[0, 0, 0]))
            object.__setattr__(self, "side_camera",     Camera(position=[0, y_side, z_side], look_at=[0, 0, 0]))
            object.__setattr__(self, "diag_camera",     Camera(position=[x_diag, y_diag, z_diag], look_at=[0, 0, 0]))

    # READ access: if the attribute isn't on the wrapper, fetch it from the inner Scene.
    def __getattr__(self, name):
        return getattr(self._scene, name)

    # WRITE/SET access: forward to inner Scene unless it's a wrapper-owned field
    def __setattr__(self, name, value):
        # during early init, _scene may not exist yet
        if name in MyScene._OWN_FIELDS or name.startswith("_") or "_scene" not in self.__dict__:
            object.__setattr__(self, name, value)
        elif hasattr(self._scene, name):
            setattr(self._scene, name, value)
        else:
            # allow creating new wrapper fields if you want; or raise to forbid
            object.__setattr__(self, name, value)

    # DELETE access: mirror the same logic
    def __delattr__(self, name):
        if name in MyScene._OWN_FIELDS:
            object.__delattr__(self, name)
        elif hasattr(self._scene, name):
            delattr(self._scene, name)
        else:
            object.__delattr__(self, name)

    # Make tab completion/introspection nice: merge wrapper + inner scene attributes
    def __dir__(self):
        return sorted(set(list(self.__dict__.keys()) + list(type(self).__dict__.keys()) + dir(self._scene)))

    # (optional) expose the inner scene explicitly if you ever need it
    @property
    def scene(self):
        return self._scene
