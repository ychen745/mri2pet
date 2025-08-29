import random
import scipy
import numpy as np
import SimpleITK as sitk


class Transform(object):
    def __init__(self, target='image',
                 interpolator_image=sitk.sitkLinear,
                 interpolator_label=sitk.sitkLinear,
                 _interpolator_image='linear',
                 _interpolator_label='linear',
                 ):
        self.target = target
        self.interpolator_image = interpolator_image
        self.interpolator_label = interpolator_label
        self._interpolator_image = _interpolator_image
        self._interpolator_label = _interpolator_label

    def resample(self, image):
        reference_image = image
        interpolator = self.interpolator_image if self.target == 'image' else self.interpolator_label
        default_value = 0
        return sitk.Resample(image, reference_image, self.transform, interpolator, default_value)

    def get_center(self, image):
        width, height, depth = image.GetSize()
        return image.TransformIndexToPhysicalPoint((int(np.ceil(width / 2)),
                                                  int(np.ceil(height / 2)),
                                                  int(np.ceil(depth / 2))))

    def rotation3d(self, image, theta_x, theta_y, theta_z):
        """
        This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
        respectively
        :param image: An sitk MRI image
        :param theta_x: The amount of degrees the user wants the image rotated around the x axis
        :param theta_y: The amount of degrees the user wants the image rotated around the y axis
        :param theta_z: The amount of degrees the user wants the image rotated around the z axis
        :param show: Boolean, whether or not the user wants to see the result of the rotation
        :return: The rotated image
        """
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
        image_center = self.get_center(image)
        euler_transform.SetCenter(image_center)
        euler_transform.SetRotation(theta_x, theta_y, theta_z)
        resampled_image = self.resample(image, euler_transform)
        return resampled_image

    def flipit(self, image, axes):
        array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        if axes == 0:
            array = np.fliplr(array)
        if axes == 1:
            array = np.flipud(array)

        img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
        img.SetDirection(direction)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)

        return image

    def brightness(self, image):
        array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        max = 255
        min = 0

        c = np.random.randint(-20, 20)

        array = array + c

        array[array >= max] = max
        array[array <= min] = min

        img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
        img.SetDirection(direction)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)

        return img

    def contrast(self, image):
        array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        shape = array.shape
        ntotpixel = shape[0] * shape[1] * shape[2]
        IOD = np.sum(array)
        luminanza = int(IOD / ntotpixel)

        c = np.random.randint(-20, 20)

        d = array - luminanza
        dc = d * abs(c) / 100

        if c >= 0:
            J = array + dc
            J[J >= 255] = 255
            J[J <= 0] = 0
        else:
            J = array - dc
            J[J >= 255] = 255
            J[J <= 0] = 0

        img = sitk.GetImageFromArray(np.transpose(J, axes=(2, 1, 0)))
        img.SetDirection(direction)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)

        return img

    def translateit(self, image, offset, isseg=False):
        order = 0 if isseg == True else 5

        array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        array = scipy.ndimage.interpolation.shift(array, (int(offset[0]), int(offset[1]), 0), order=order)

        img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
        img.SetDirection(direction)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)

        return img

    def imadjust(self, image,gamma=np.random.uniform(1, 2)):

        array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        array = (((array - array.min()) / (array.max() - array.min())) ** gamma) * (255 - 0) + 0

        img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
        img.SetDirection(direction)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)

        return img


class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
      Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
      This transformation only applicable in train mode

    Args:
      output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

    def __init__(self, output_size=64, drop_ratio=0.1, min_pixel=1):
        self.name = 'Random Crop'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(drop_ratio, (int, float))
        if drop_ratio >= 0 and drop_ratio <= 1:
            self.drop_ratio = drop_ratio
        else:
            raise RuntimeError('Drop ratio should be between 0 and 1')

        assert isinstance(min_pixel, int)
        if min_pixel >= 0:
            self.min_pixel = min_pixel
        else:
            raise RuntimeError('Min label pixel count should be integer larger than 0')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        size_new = self.output_size

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        while not contain_label:
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0] - size_new[0])

            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1] - size_new[1])

            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2] - size_new[2])

            roiFilter.SetIndex([start_i, start_j, start_k])

            # threshold label into only ones and zero
            threshold = sitk.BinaryThresholdImageFilter()
            threshold.SetLowerThreshold(1)
            threshold.SetUpperThreshold(255)
            threshold.SetInsideValue(1)
            threshold.SetOutsideValue(0)
            mask = threshold.Execute(label)
            mask_cropped = roiFilter.Execute(mask)
            label_crop = roiFilter.Execute(label)
            statFilter = sitk.StatisticsImageFilter()
            statFilter.Execute(mask_cropped)  # mine for GANs

            # will iterate until a sub volume containing label is extracted
            # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
            # if statFilter.GetSum()/pixel_count<self.min_ratio:
            if statFilter.GetSum() < self.min_pixel:
                contain_label = self.drop(self.drop_ratio)  # has some probabilty to contain patch with empty label
            else:
                contain_label = True

        image_crop = roiFilter.Execute(image)

        return {'image': image_crop, 'label': label_crop, 'index': (start_i, start_j, start_k)}

    def drop(self, probability):
        return random.random() <= probability


class Augmentation(object):
    """
    Application of transforms. This is usually used for data augmentation.
    List of transforms: random noise
    """
    def __init__(self):
        self.name = 'Augmentation'
        self.image_transform = Transform(target='image')
        self.label_transform = Transform(target='label')

    def __call__(self, sample):
        """
        Choices:
            0. No augmentation
            1. Additive Gaussian Noise
            2. Recursive Gaussian
            3. Random rotation x y z
            4. BSpline Deformation
            5. Random flip
            6. Brightness
            7. Contrast
            8. Translate
            9. Random rotation z
            10. Random rotation x
            11. Random rotation y
            12. histogram gamma
        """
        # choice = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        choice = np.random.choice([0, 1, 2, 3, 4, 5])

        # no augmentation
        if choice == 0:  # no augmentation
            image, label = sample['image'], sample['label']
            return {'image': image, 'label': label}

        # Additive Gaussian noise
        if choice == 1:  # Additive Gaussian noise
            mean = np.random.uniform(0, 1)
            std = np.random.uniform(0, 2)
            self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
            self.noiseFilter.SetMean(mean)
            self.noiseFilter.SetStandardDeviation(std)

            image, label = sample['image'], sample['label']
            image = self.noiseFilter.Execute(image)
            # label = self.noiseFilter.Execute(label)

            return {'image': image, 'label': label}

        # Recursive Gaussian
        if choice == 2:  # Recursive Gaussian
            sigma = np.random.uniform(0, 1.5)
            self.noiseFilter = sitk.RecursiveGaussianImageFilter()
            self.noiseFilter.SetOrder(0)
            self.noiseFilter.SetSigma(sigma)

            image, label = sample['image'], sample['label']
            image = self.noiseFilter.Execute(image)
            # label = self.noiseFilter.Execute(label)

            return {'image': image, 'label': label}

        # Random rotation x y z
        if choice == 3:  # Random rotation

            theta_x = np.random.randint(-40, 40)
            theta_y = np.random.randint(-40, 40)
            theta_z = np.random.randint(-180, 180)
            image, label = sample['image'], sample['label']

            image = self.image_transform.rotation3d(image, theta_x,theta_y, theta_z)
            label = self.label_transform.rotation3d(label, theta_x,theta_y, theta_z)

            return {'image': image, 'label': label}

        # BSpline Deformation
        if choice == 4:  # BSpline Deformation

            randomness = 10

            assert isinstance(randomness, (int, float))
            if randomness > 0:
                self.randomness = randomness
            else:
                raise RuntimeError('Randomness should be non zero values')

            image, label = sample['image'], sample['label']
            spline_order = 3
            domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                          image.GetSize()[1] * image.GetSpacing()[1],
                                          image.GetSize()[2] * image.GetSpacing()[2]]

            bspline = sitk.BSplineTransform(3, spline_order)
            bspline.SetTransformDomainOrigin(image.GetOrigin())
            bspline.SetTransformDomainDirection(image.GetDirection())
            bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
            bspline.SetTransformDomainMeshSize((10, 10, 10))

            # Random displacement of the control points.
            originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * self.randomness
            bspline.SetParameters(originalControlPointDisplacements)

            image = sitk.Resample(image, bspline)
            label = sitk.Resample(label, bspline)
            return {'image': image, 'label': label}

        # Random flip
        if choice == 5:  # Random flip

            axes = np.random.choice([0, 1])
            image, label = sample['image'], sample['label']

            image = self.image_transform.flipit(image, axes)
            label = self.label_transform.flipit(label, axes)

            return {'image': image, 'label': label}

        # Brightness
        if choice == 6:  # Brightness

            image, label = sample['image'], sample['label']

            image = self.image_transform.brightness(image)
            # label = self.label_transform.brightness(label)

            return {'image': image, 'label': label}

        # Contrast
        if choice == 7:  # Contrast

            image, label = sample['image'], sample['label']

            image = self.image_transform.contrast(image)
            # label = self.label_transform.contrast(label)

            return {'image': image, 'label': label}

        # Translate
        if choice == 8:  # translate

            image, label = sample['image'], sample['label']

            t1 = np.random.randint(-40, 40)
            t2 = np.random.randint(-40, 40)
            offset = [t1, t2]

            image = self.image_transform.translateit(image, offset)
            label = self.label_transform.translateit(label, offset)

            return {'image': image, 'label': label}

        # Random rotation z
        if choice == 9:  # Random rotation

            theta_x = 0
            theta_y = 0
            theta_z = np.random.randint(-180, 180)
            image, label = sample['image'], sample['label']

            image = self.image_transform.rotation3d(image, theta_x, theta_y, theta_z)
            label = self.label_transform.rotation3d(label, theta_x, theta_y, theta_z)

            return {'image': image, 'label': label}

        # Random rotation x
        if choice == 10:  # Random rotation

            theta_x = np.random.randint(-40, 40)
            theta_y = 0
            theta_z = 0
            image, label = sample['image'], sample['label']

            image = self.image_transform.rotation3d(image, theta_x, theta_y, theta_z)
            label = self.label_transform.rotation3d(label, theta_x, theta_y, theta_z)

            return {'image': image, 'label': label}

        # Random rotation y
        if choice == 11:  # Random rotation

            theta_x = 0
            theta_y = np.random.randint(-40, 40)
            theta_z = 0
            image, label = sample['image'], sample['label']

            image = self.image_transform.rotation3d(image, theta_x, theta_y, theta_z)
            label = self.label_transform.rotation3d(label, theta_x, theta_y, theta_z)

            return {'image': image, 'label': label}

        # histogram gamma
        if choice == 12:
            image, label = sample['image'], sample['label']

            image = self.image_transform.imadjust(image)

            return {'image': image, 'label': label}


