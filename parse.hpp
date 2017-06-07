
class ConfigFile {
private:
  std::map<std::string, std::string> contents;
  std::string fName;

public:
  ConfigFile(const std::string &fName);
  bool keyExists(const std::string &key) const;
  template <typename ValueType>
  ValueType getValueOfKey(const std::string &key, ValueType const &defaultValue = ValueType()) const;
};
