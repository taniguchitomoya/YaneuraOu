// NNUE�]���֐��̊w�K�N���X�e���v���[�g��Dropout�p���ꉻ

// C++��NNUE��f�B�[�v���[�j���O�ɂ��Đ[�������̂Ȃ��܂܃R�s�y�ŏ������̂�
// �ԈႢ���܂މ\��������̂͂������̂��ƁA�R�s�y���R���̕K�v�̂Ȃ��R�[�h�⑫��ĂȂ����������邩������Ȃ�

#ifndef _NNUE_TRAINER_DROPOUT_H_
#define _NNUE_TRAINER_DROPOUT_H_

#include "../../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include "../../../learn/learn.h"
#include "../layers/dropout.h"
#include "trainer.h"

#include <random>

namespace Eval {

	namespace NNUE {

		template <typename PreviousLayer>
		class Trainer<Layers::Dropout<PreviousLayer>> {
		private:
			// �w�K�Ώۂ̑w�̌^
			using LayerType = Layers::Dropout<PreviousLayer>;
			//random.�����P�ƂŎ��̂ł͂Ȃ��A���̏ꏊ�Ƌ��p���������ǂ���������Ȃ�
			std::mt19937 mtrandom;

			//dropout�̃t���O�ۑ��p
			bool dropout_mask[LayerType::kOutputDimensions];


		public:
			// �t�@�N�g���֐�
			static std::shared_ptr<Trainer> Create(
				LayerType* target_layer, FeatureTransformer* feature_transformer) {
				return std::shared_ptr<Trainer>(
					new Trainer(target_layer, feature_transformer));
			}

			// �n�C�p�[�p�����[�^�Ȃǂ̃I�v�V������ݒ肷��
			void SendMessage(Message* message) {
				previous_layer_trainer_->SendMessage(message);
			}

			// �p�����[�^�𗐐��ŏ���������
			template <typename RNG>
			void Initialize(RNG& rng) {
				previous_layer_trainer_->Initialize(rng);
			}

			// ���`�d
			const LearnFloatType* Propagate(const std::vector<Example>& batch) {
				if (output_.size() < kOutputDimensions * batch.size()) {
					output_.resize(kOutputDimensions * batch.size());
					gradients_.resize(kInputDimensions * batch.size());
				}
				const auto input = previous_layer_trainer_->Propagate(batch);

				std::uniform_real_distribution<double> zeroone(0.0, 1.0);

				for (IndexType i = 0; i < kOutputDimensions; ++i) {
					dropout_mask[i] = zeroone(mtrandom) < 0.5;
				}

				batch_size_ = static_cast<IndexType>(batch.size());
				for (IndexType b = 0; b < batch_size_; ++b) {
					const IndexType batch_offset = kOutputDimensions * b;
					for (IndexType i = 0; i < kOutputDimensions; ++i) {
						const IndexType index = batch_offset + i;
						output_[index] = input[index] * dropout_mask[i];
					}
				}
				return output_.data();
			}

			// �t�`�d
			void Backpropagate(const LearnFloatType* gradients,
				LearnFloatType learning_rate) {
				for (IndexType b = 0; b < batch_size_; ++b) {
					const IndexType batch_offset = kOutputDimensions * b;
					for (IndexType i = 0; i < kOutputDimensions; ++i) {
						const IndexType index = batch_offset + i;
						gradients_[index] = gradients[index] * dropout_mask[i];
					}
				}
				previous_layer_trainer_->Backpropagate(gradients_.data(), learning_rate);
			}

		private:
			// �R���X�g���N�^
			Trainer(LayerType* target_layer, FeatureTransformer* feature_transformer) :
				batch_size_(0),
				previous_layer_trainer_(Trainer<PreviousLayer>::Create(
					&target_layer->previous_layer_, feature_transformer)),
				target_layer_(target_layer) {
			}

			// ���o�͂̎�����
			static constexpr IndexType kInputDimensions = LayerType::kOutputDimensions;
			static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;


			// �~�j�o�b�`�̃T���v����
			IndexType batch_size_;

			// ���O�̑w��Trainer
			const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

			// �w�K�Ώۂ̑w
			LayerType* const target_layer_;

			// ���`�d�p�o�b�t�@
			std::vector<LearnFloatType> output_;

			// �t�`�d�p�o�b�t�@
			std::vector<LearnFloatType> gradients_;

		};

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)

#endif
