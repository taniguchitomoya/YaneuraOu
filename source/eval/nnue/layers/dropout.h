// NNUE�]���֐��̑wDropout�̒�`

// C++��NNUE��f�B�[�v���[�j���O�ɂ��Đ[�������̂Ȃ��܂܃R�s�y�ŏ������̂�
// �ԈႢ���܂މ\��������̂͂������̂��ƁA�R�s�y���R���̕K�v�̂Ȃ��R�[�h�⑫��ĂȂ����������邩������Ȃ�


#ifndef _NNUE_LAYERS_DROPOUT_H_
#define _NNUE_LAYERS_DROPOUT_H_

#include "../../../shogi.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace Eval {

	namespace NNUE {

		namespace Layers {

			// Dropout
			template <typename PreviousLayer>
			class Dropout {
			public:
				// ���o�͂̌^
				using InputType = typename PreviousLayer::OutputType;
				using OutputType = InputType;
				//static_assert(std::is_same<InputType, std::int32_t>::value, "");

				// ���o�͂̎�����
				static constexpr IndexType kInputDimensions =
					PreviousLayer::kOutputDimensions;
				static constexpr IndexType kOutputDimensions = kInputDimensions;

				// ���̑w�Ŏg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
				static constexpr std::size_t kSelfBufferSize =
					CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

				// ���͑w���炱�̑w�܂łŎg�p���鏇�`�d�p�o�b�t�@�̃T�C�Y
				static constexpr std::size_t kBufferSize =
					PreviousLayer::kBufferSize + kSelfBufferSize;

				// �]���֐��t�@�C���ɖ��ߍ��ރn�b�V���l
				static constexpr std::uint32_t GetHashValue() {
					//�]���֐��t�@�C����tnk_wcsc28�݊��Ƃ��邽�߁A������hash�l��ύX���Ȃ�
					return PreviousLayer::GetHashValue();

					/*
					std::uint32_t hash_value = 0x0DDD1234u;
					hash_value += PreviousLayer::GetHashValue();
					return hash_value;
					*/
				}

				// ���͑w���炱�̑w�܂ł̍\����\��������
				static std::string GetStructureString() {
					//�]���֐��t�@�C����tnk_wcsc28�݊��Ƃ��邽�߁A�����ĕ������ύX���Ȃ�
					return PreviousLayer::GetStructureString();
					/*
					return "Dropout[" +
						std::to_string(kOutputDimensions) + "](" +
						PreviousLayer::GetStructureString() + ")";
					*/
				}

				// �p�����[�^��ǂݍ���
				bool ReadParameters(std::istream& stream) {
					return previous_layer_.ReadParameters(stream);
				}

				// �p�����[�^����������
				bool WriteParameters(std::ostream& stream) const {
					return previous_layer_.WriteParameters(stream);
				}

				// ���`�d
				const OutputType* Propagate(
					const TransformedFeatureType* transformed_features, char* buffer) const {
					const auto input = previous_layer_.Propagate(
						transformed_features, buffer + kSelfBufferSize);
					const auto output = reinterpret_cast<OutputType*>(buffer);

					for (IndexType i = 0; i < kOutputDimensions; ++i) {
						output[i] = input[i] / 2; //output[i] * ( 1 -  dropout ratio(0.5) )
					}
					return output;
				}

			private:
				// �w�K�p�N���X��friend�ɂ���
				friend class Trainer<Dropout>;

				// ���̑w�̒��O�̑w
				PreviousLayer previous_layer_;
			};

		}  // namespace Layers

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
