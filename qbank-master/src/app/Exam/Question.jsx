import React from 'react';

const Question = ({ question, options, selectedOption, onChangeOption }) => {
  return (
    <div className="border p-4 mb-4">
      <p className="font-semibold">{question}</p>
      <div className="mt-2">
        {options.map((option, index) => (
          <div key={index} className="flex items-center">
            <input
              type="radio"
              id={`option${index}`}
              name="mcqOption"
              value={index}
              checked={selectedOption === index}
              onChange={() => onChangeOption(index)}
              className="mr-2"
            />
            <label htmlFor={`option${index}`}>{option}</label>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Question;