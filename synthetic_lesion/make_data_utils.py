from monai.transforms import MapTransform

# Define a custom DictionaryTransform


class MakeLesionMaskedDatad(MapTransform):

    def __call__(self, data):
        # Define the operation you want to perform on 'image'
        orig_img = data['image'][[1]]
        data['image'] = data['image'][[1]] * (data['label'][[1]] < 0.5)
        data['label'] = orig_img
        #data['image'] = data['image'] * data['label'][2]
        return data
