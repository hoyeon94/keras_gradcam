# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		for layer in reversed(self.model.layers):
			if len(layer.output_shape) == 4:
				return layer.name
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")